import re
import matplotlib.pyplot as plt

def parse_logs(log_file):
    epoch_pattern = re.compile(r'Epoch (\d+/\d+)')
    loss_pattern = re.compile(r'loss 2d: (\d+\.\d+), loss 3d: ([\d\.]+), mesh loss 3d: ([\d\.]+), photometric loss: ([\d\.]+)')
    val_loss_pattern = re.compile(r'val loss 2d: (\d+\.\d+), val loss 3d: ([\d\.]+), val mesh loss 3d: ([\d\.]+), val photometric loss: ([\d\.]+)')
    
    epochs = []
    losses_2d = []
    losses_3d = []
    mesh_losses_3d = []
    photometric_losses = []
    val_losses_2d = []
    val_losses_3d = []
    val_mesh_losses_3d = []
    val_photometric_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            loss_match = loss_pattern.search(line)
            val_loss_match = val_loss_pattern.search(line)
            
            if epoch_match:
                epochs.append(epoch_match.group(1))
                
            if loss_match:
                losses_2d.append(float(loss_match.group(1)))
                losses_3d.append(float(loss_match.group(2)))
                mesh_losses_3d.append(float(loss_match.group(3)))
                photometric_losses.append(float(loss_match.group(4)))
                
            if val_loss_match:
                val_losses_2d.append(float(val_loss_match.group(1)))
                val_losses_3d.append(float(val_loss_match.group(2)))
                val_mesh_losses_3d.append(float(val_loss_match.group(3)))
                val_photometric_losses.append(float(val_loss_match.group(4)))
                
    return epochs, losses_2d, losses_3d, mesh_losses_3d, photometric_losses, val_losses_2d, val_losses_3d, val_mesh_losses_3d, val_photometric_losses

def plot_losses(log_file):
    epochs, losses_2d, losses_3d, mesh_losses_3d, photometric_losses, val_losses_2d, val_losses_3d, val_mesh_losses_3d, val_photometric_losses = parse_logs(log_file)
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, losses_2d, label='Loss 2D', marker='o')
    plt.plot(epochs, losses_3d, label='Loss 3D', marker='o')
    plt.plot(epochs, mesh_losses_3d, label='Mesh Loss 3D', marker='o')
    plt.plot(epochs, photometric_losses, label='Photometric Loss', marker='o')
    plt.plot(epochs, val_losses_2d, label='Validation Loss 2D', marker='x')
    plt.plot(epochs, val_losses_3d, label='Validation Loss 3D', marker='x')
    plt.plot(epochs, val_mesh_losses_3d, label='Validation Mesh Loss 3D', marker='x')
    plt.plot(epochs, val_photometric_losses, label='Validation Photometric Loss', marker='x')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage
log_file = 'path_to_your_log_file.log'
plot_losses(log_file)