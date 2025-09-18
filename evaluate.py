import torch

def evaluate(model, val_loader, classification_criterion, device):
    """
    Evaluates the model on the validation dataset.
    
    This function is crucial for early stopping, as it provides the metric
    (validation loss) to monitor for improvement.
    """
    model.eval() # Set the model to evaluation mode
    running_val_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # We don't need to calculate gradients during evaluation
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass - we don't need the metrics for evaluation
            # Passing is_training=False tells the MoE layer to skip aux loss calculation
            logits, _ = model(images, is_training=False)
            
            # Calculate classification loss only
            loss = classification_criterion(logits, labels)
            running_val_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

    val_loss = running_val_loss / len(val_loader)
    val_acc = total_correct / total_samples
    
    return val_loss, val_acc