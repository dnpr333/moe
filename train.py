import torch
import torch.nn as nn
from tqdm import tqdm # For a nice progress bar

def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    classification_criterion,
    config,
    device: torch.device,
):
    """
    Train ViT-MoE for one epoch.

    If the ViTMOE submodules expose an attribute `auxiliary_loss` (e.g. a scalar
    collected from all SparseMoE layers), it will be added to the total loss
    using the weight given in config['aux_loss_weight'] (default 0.1).
    """
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    aux_loss_weight = 0.1

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)               
        logits  = outputs.logits

        # Main classification loss
        cls_loss = classification_criterion(logits, labels)

        # Optional MoE auxiliary loss (if present)
        aux_loss = getattr(outputs, "auxiliary_loss", 0.0)
        if not torch.is_tensor(aux_loss):
            aux_loss = torch.tensor(aux_loss, device=logits.device, dtype=logits.dtype)

        total_loss = cls_loss + config.get("aux_loss_weight", 0.1) * aux_loss

        # Back-prop
        total_loss.backward()
        optimizer.step()

        # Stats
        running_loss += total_loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        pbar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            cls=f"{cls_loss.item():.4f}",
            aux=f"{float(aux_loss):.4f}" if isinstance(aux_loss, torch.Tensor) else f"{aux_loss:.4f}",
        )

    epoch_loss = running_loss / total_samples
    epoch_acc  = total_correct / total_samples
    return epoch_loss, epoch_acc