import torch
from tqdm import tqdm
import torch.nn as nn
def evaluate(model: nn.Module, val_loader, classification_criterion, device: torch.device):
    """
    Evaluate ViT-MoE on a validation set.

    Returns:
        val_loss (float): average cross-entropy loss
        val_acc  (float): accuracy
    """
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating", leave=False)
        for batch in pbar:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            # Forward pass – HuggingFace models return an output object
            outputs = model(images)
            logits  = outputs.logits

            loss = classification_criterion(logits, labels)
            running_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    val_loss = running_loss / total_samples
    val_acc  = total_correct / total_samples
    return val_loss, val_acc