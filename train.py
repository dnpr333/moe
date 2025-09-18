import torch
import torch.nn as nn
from tqdm import tqdm # For a nice progress bar

def train_one_epoch(model, train_loader, optimizer, classification_criterion, config, device):
    """
    Trains the model for one epoch.
    
    This function implements the core training logic, including the calculation
    of the combined loss function as specified in the V-MoE paper [2].
    """
    model.train() # Set the model to training mode
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Get the auxiliary loss weight from the config [2]
    aux_loss_weight = config['training_config']['aux_loss_weight']
    
    # Get the capacity ratio for training [3]
    capacity_ratio = config['training_config']['capacity_ratio']

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # --- Forward Pass ---
        # The model should return logits for classification and a metrics dict
        # containing the auxiliary loss from the MoE layers.
        logits, metrics = model(images, capacity_ratio=capacity_ratio, is_training=True)
        
        # --- Loss Calculation ---
        # 1. Calculate the primary classification loss
        classification_loss = classification_criterion(logits, labels)
        
        # 2. Get the auxiliary loss for load balancing from the MoE layers [1, 2]
        # If you have multiple MoE layers, you might need to average their aux losses.
        # This implementation assumes the model already handles that and returns a single value.
        aux_loss = metrics.get('l_aux', 0.0) # Default to 0 if no MoE layer was used
        
        # 3. Combine the losses using the weight λ (lambda) [2]
        total_loss = classification_loss + aux_loss_weight * aux_loss
        
        # --- Backward Pass and Optimization ---
        total_loss.backward()
        optimizer.step()
        
        # --- Statistics ---
        running_loss += total_loss.item()
        
        _, predicted = torch.max(logits.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'cls_loss': f"{classification_loss.item():.4f}",
            'aux_loss': f"{aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss:.4f}"
        })
        
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = total_correct / total_samples
    
    return epoch_loss, epoch_acc