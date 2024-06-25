"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union
import os

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform, multiframe):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False
        self.multiframe = multiframe

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))
        if self.multiframe and 'prev_frames' in images:
            prev_frames = images['prev_frames']
            images = images['inputs']
            images_mf = [[img] + prev_frame for img, prev_frame in zip(images, prev_frames)]
            
            all_images = [frame for sublist in images_mf for frame in sublist]
            
            original_image_sizes: List[Tuple[int, int]] = []
            for img in all_images:
                val = img.shape[-2:]
                assert len(val) == 2
                original_image_sizes.append((val[0], val[1]))
            
            original_images = [[frame.permute(1, 2, 0) for frame in sublist] for sublist in images_mf]
            
            # adapt targets
            targets_mf = []
            for i in range(len(targets)):
                target = targets[i]
                new_target = [target for _ in range(len(images_mf[i]))]
                targets_mf.append(new_target)
            
            for i, (images_sample, targets_sample) in enumerate(zip(images_mf, targets_mf)):
                images_sample, targets_sample = self.transform(images_sample, targets_sample)
                images_mf[i], targets_mf[i] = images_sample, targets_sample
        else: # NO self.multiframe
            images = images['inputs']
            
            original_image_sizes: List[Tuple[int, int]] = []
            for img in images:
                val = img.shape[-2:]
                assert len(val) == 2
                original_image_sizes.append((val[0], val[1]))
            
            original_images = [img.permute(1, 2, 0) for img in images]
            
            images, targets = self.transform(images, targets)
            
        # old_targets = targets
        

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None and not self.multiframe:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))
        if self.multiframe:
            features_list = []
            for images_sample in images_mf:
                features = self.backbone(images_sample.tensors)
                for key in features:
                    features[key] = torch.mean(features[key], dim=0, keepdim=True)
                features_list.append(features)
            features = OrderedDict()
            for key in features_list[0].keys():
                tensors = [feature_dict[key] for feature_dict in features_list]
                concatenated_tensor = torch.cat(tensors, dim=0)
                features[key] = concatenated_tensor
        else:
            features = self.backbone(images.tensors) # extract image features
        # if self.multiframe and prev_frames:
        #     reshaped_list = []
        #     for i in range(len(prev_frames)):
        #         for j in range(len(prev_frames[i])):
        #             p = prev_frames[i][j]
        #             p, _ = self.transform([p], old_targets)
        #             prev_frames[i] = p
        #     for i in range(len(prev_frames[0])):
        #         combined_tensor = torch.stack([tensor[i] for tensor in prev_frames])
        #         reshaped_list.append(combined_tensor)
        #     for i in range(len(reshaped_list)):
        #         features_prev_frame = self.backbone(reshaped_list[i]) 
        #         for key in features.keys():
        #             features[key] = torch.cat((features[key], features_prev_frame[key]), dim=3)
            
            # for i in range(len(prev_frames[0])):
            #     combined_tensor_list = []
            #     for pfs in prev_frames:
            #         pfs, _ = self.transform(pfs, None)
            #         combined_tensor_list.append(pfs.tensors[i])
                
            #     combined_tensor = torch.stack(combined_tensor_list)
            #     features_prev_frame = self.backbone(combined_tensor)
            #     features = torch.cat((features, features_prev_frame), dim=0)

            #     # Clear memory for tensors that are no longer needed
            #     del combined_tensor_list
            #     del combined_tensor
            #     del features_prev_frame
            #     torch.cuda.empty_cache()  # if using GPU   
                 
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, original_images, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
        return losses, detections
