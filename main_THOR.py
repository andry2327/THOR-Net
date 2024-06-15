# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

""" import libraries"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import os
import datetime
import pytz
from tqdm import tqdm

from utils.options import parse_args_function
from utils.utils import freeze_component, calculate_keypoints, create_loader
# for H2O dataset only
# from utils.h2o_utils.h2o_dataset_utils import load_tar_split 
# from utils.h2o_utils.h2o_preprocessing_utils import MyPreprocessor

from models.thor_net import create_thor

torch.multiprocessing.set_sharing_strategy('file_system')

args = parse_args_function()
output_folder = args.output_file.rpartition(os.sep)[0]
# print(f'args:')
# for arg, value in vars(args).items():
#     print(f"{arg}: {value}", end=' | ')
# print('-'*30)

# DEBUG
# args.dataset_name = 'povsurgery' 
# args.root = '/content/drive/MyDrive/Thesis/THOR-Net_based_work/povsurgery/object_False' 
# args.output_file = '/content/drive/MyDrive/Thesis/THOR-Net_based_work/checkpoints/THOR-Net_trained_on_POV-Surgery_object_False/Training--14-06-2024_10-53/model-' 
# output_folder = args.output_file.rpartition(os.sep)[0]
# args.batch_size = 1
# args.num_iteration = 3
# args.object = False 
# args.pretrained_model=''#'/content/drive/MyDrive/Thesis/THOR-Net_based_work/checkpoints/THOR-Net_trained_on_POV-Surgery_object_False/Training--14-06-2024_10-53/model-1.pkl'
# args.hands_connectivity_type = 'simple'

# Define device
device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

num_kps2d, num_kps3d, num_verts = calculate_keypoints(args.dataset_name, args.object)

""" Configure a log """

files_in_dir = os.listdir(output_folder)
log_file = [x for x in files_in_dir if x.endswith('.txt')]

if log_file:
    # If there is an existing log file, use the first one found
    filename_log = os.path.join(output_folder, log_file[0])
else:
    # Create a new log file
    filename_log = os.path.join(
        output_folder,
        f'log_{output_folder.rpartition(os.sep)[-1]}.txt'
    )

# Configure the logging
log_format = '%(message)s'
logging.basicConfig(
    filename=filename_log,
    level=logging.INFO,
    format=log_format,
    filemode='a'  
)
fh = logging.FileHandler(filename_log, mode='a')  # Use 'a' mode to append to the log file
fh.setFormatter(logging.Formatter(log_format))
logger = logging.getLogger(__name__)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

logging.info(f'args:') if not log_file else None
print(f'args:')
for arg, value in vars(args).items():
    logging.info(f"--{arg}: {value}") if not log_file else None
    print(f"{arg}: {value}", end=' | ')
logging.info('--'*50) if not log_file else None
print('\n')

print(f'🟢 Logging info in "{filename_log}"')

""" load datasets """

if args.dataset_name.lower() == 'h2o':
    annotation_components = ['cam_pose', 'hand_pose', 'hand_pose_mano', 'obj_pose', 'obj_pose_rt', 'action_label', 'verb_label']
    my_preprocessor = MyPreprocessor('../mano_v1_2/models/', 'datasets/objects/mesh_1000/', args.root)
    h2o_data_dir = os.path.join(args.root, 'shards')
    train_input_tar_lists, train_annotation_tar_files = load_tar_split(h2o_data_dir, 'train')    
    val_input_tar_lists, val_annotation_tar_files = load_tar_split(h2o_data_dir, 'val')   
    num_classes = 4
    graph_input = 'coords'
else: # i.e. HO3D, POV-Surgery
    print(f'Loading training data ...', end=' ')
    trainloader = create_loader(args.dataset_name, args.root, 'train', batch_size=args.batch_size, num_kps3d=num_kps3d, num_verts=num_verts, is_sample_dataset=True)
    print(f'✅ Training data loaded.')
    print(f'Loading validation data ...', end=' ')
    valloader = create_loader(args.dataset_name, args.root, 'val', batch_size=args.batch_size, is_sample_dataset=True)
    print(f'✅ Validation data loaded.')
    num_classes = 2 
    graph_input = 'heatmaps'

""" load model """
model = create_thor(num_kps2d=num_kps2d, num_kps3d=num_kps3d, num_verts=num_verts, num_classes=num_classes, 
                                rpn_post_nms_top_n_train=num_classes-1, 
                                device=device, num_features=args.num_features, hid_size=args.hid_size,
                                photometric=args.photometric, graph_input=graph_input, dataset_name=args.dataset_name, testing=args.testing,
                                hands_connectivity_type=args.hands_connectivity_type)

print('🟢 THOR-Net is loaded')

if torch.cuda.is_available():
    model = model.cuda(args.gpu_number[0])
    model = nn.DataParallel(model, device_ids=args.gpu_number)

""" load saved model"""

if args.pretrained_model != '':
    state_dict = torch.load(args.pretrained_model, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except:
        for key in list(state_dict.keys()):
            state_dict[key.replace('module.', '')] = state_dict.pop(key)
        model.load_state_dict(state_dict)
    losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
    start = len(losses)
    print(f'🟢 Model checkpoint "{args.pretrained_model}" loaded')
else:
    losses = []
    start = 0

"""define optimizer"""

criterion = nn.MSELoss()
# criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
scheduler.last_epoch = start

keys = ['boxes', 'labels', 'keypoints', 'keypoints3d', 'mesh3d', 'palm']

""" training """

print('🟢 Begin training the network')

for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
    
    train_loss2d = 0.0
    running_loss2d = 0.0
    running_loss3d = 0.0
    running_mesh_loss3d = 0.0
    running_photometric_loss = 0.0
    
    if 'h2o' in args.dataset_name.lower():
        h2o_info = (train_input_tar_lists, train_annotation_tar_files, annotation_components, args.buffer_size, my_preprocessor)
        trainloader = create_loader(args.dataset_name, h2o_data_dir, 'train', args.batch_size, h2o_info=h2o_info)

    pbar = tqdm(desc=f'Epoch {epoch+1} - train: ', total=len(trainloader))
    for i, tr_data in enumerate(trainloader):
        
        # get the inputs
        data_dict = tr_data
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward
        targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
        inputs = [t['inputs'].to(device) for t in data_dict]
        loss_dict = model(inputs, targets)
        
        # Calculate Loss
        loss = sum(loss for _, loss in loss_dict.items())
        
        # Backpropagate
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss2d += loss_dict['loss_keypoint'].data
        running_loss2d += loss_dict['loss_keypoint'].data
        running_loss3d += loss_dict['loss_keypoint3d'].data
        running_mesh_loss3d += loss_dict['loss_mesh3d'].data
        if 'loss_photometric' in loss_dict.keys():
            running_photometric_loss += loss_dict['loss_photometric'].data

        if (i+1) % args.log_batch == 0:    # print every log_iter mini-batches
            logging.info('[Epoch %d/%d, Processed data %d/%d] loss 2d: %.4f, loss 3d: %.4f, mesh loss 3d: %.4f, photometric loss: %.4f' % 
            (epoch + 1, args.num_iterations, i + 1, len(trainloader), running_loss2d / args.log_batch, running_loss3d / args.log_batch, 
            running_mesh_loss3d / args.log_batch, running_photometric_loss / args.log_batch))
            running_mesh_loss3d = 0.0
            running_loss2d = 0.0
            running_loss3d = 0.0
            running_photometric_loss = 0.0
            
        pbar.update(1)
    pbar.close()
    
    losses.append((train_loss2d / (i+1)).cpu().numpy())
    
    if (epoch+1) % args.snapshot_epoch == 0:
        torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
        np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))
        # delete files from older epochs
        if epoch+1 > 1:
            files_to_delete = [x for x in os.listdir(args.output_folder) if f'model-{epoch}' in x]
            for file in files_to_delete:
                try:
                    os.remove(os.join(args.output_folder, file))
                except:
                    pass
        print(f'Model checkpoint (epoch {epoch+1}) saved in "{args.output_file}"')

    if (epoch+1) % args.val_epoch == 0:
        val_loss2d = 0.0
        val_loss3d = 0.0
        val_mesh_loss3d = 0.0
        val_photometric_loss = 0.0
        
        # model.module.transform.training = False
        
        if 'h2o' in args.dataset_name.lower():
            h2o_info = (val_input_tar_lists, val_annotation_tar_files, annotation_components, args.buffer_size, my_preprocessor)
            valloader = create_loader(args.dataset_name, h2o_data_dir, 'val', args.batch_size, h2o_info)

        pbar = tqdm(desc=f'Epoch {epoch+1} - val: ', total=len(valloader))
        for v, val_data in enumerate(valloader):
            
            # get the inputs
            data_dict = val_data
        
            # wrap them in Variable
            targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
            inputs = [t['inputs'].to(device) for t in data_dict]    
            loss_dict = model(inputs, targets)
            
            val_loss2d += loss_dict['loss_keypoint'].data
            val_loss3d += loss_dict['loss_keypoint3d'].data
            val_mesh_loss3d += loss_dict['loss_mesh3d'].data
            if 'loss_photometric' in loss_dict.keys():
                running_photometric_loss += loss_dict['loss_photometric'].data
            
            pbar.update(1)
        pbar.close()
        
        # model.module.transform.training = True
        
        logging.info('Epoch %d/%d - val loss 2d: %.4f, val loss 3d: %.4f, val mesh loss 3d: %.4f, val photometric loss: %.4f' % 
                    (epoch + 1, args.num_iterations, val_loss2d / (v+1), val_loss3d / (v+1), val_mesh_loss3d / (v+1), val_photometric_loss / (v+1)))  
        print('Epoch %d/%d - val loss 2d: %.4f, val loss 3d: %.4f, val mesh loss 3d: %.4f, val photometric loss: %.4f' % 
                    (epoch + 1, args.num_iterations, val_loss2d / (v+1), val_loss3d / (v+1), val_mesh_loss3d / (v+1), val_photometric_loss / (v+1)))  
    
    if args.freeze and epoch == 0:
        logging.info('Freezing Keypoint RCNN ..')            
        freeze_component(model.module.backbone)
        freeze_component(model.module.rpn)
        freeze_component(model.module.roi_heads)

    # Decay Learning Rate
    scheduler.step()

logging.info('Finished Training')