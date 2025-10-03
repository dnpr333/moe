import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModelForImageClassification
from copy import deepcopy

from transformers import ViTConfig, AutoImageProcessor, AutoModelForImageClassification
from dataloader import get_cifar100_dataloaders
from model import VisionTransformerWithMoE

def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)

            if isinstance(output, tuple) and len(output) == 2:
                logits, aux_loss = output
                aux_loss = aux_loss if aux_loss is not None else torch.tensor(0.0, device=device)
                cls_loss = criterion(logits, labels)
                loss = cls_loss + 0.01 * aux_loss
            else:
                logits = output.logits
                cls_loss = criterion(logits, labels)
                loss = cls_loss
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy, None  # third value for compatibility with previous code

def train_one_epoch(loader, model, optimizer, device):
    model.train()
    total_loss, total_cls_loss = 0.0, 0.0
    total_samples, total_correct = 0, 0
    criterion = nn.CrossEntropyLoss()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = model(imgs)

        if isinstance(output, tuple) and len(output) == 2:
            logits, aux_loss = output
            aux_loss = aux_loss if aux_loss is not None else torch.tensor(0.0, device=device)
            cls_loss = criterion(logits, labels)
            loss = cls_loss + 0.01 * aux_loss
        else:
            logits = output.logits
            cls_loss = criterion(logits, labels)
            loss = cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = 100.0 * total_correct / total_samples
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy

def train_and_evaluate(trainloader, testloader, model, num_epochs, lr, checkpoint_dir, device):
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(trainloader, model, optimizer, device)
        val_loss, val_acc, _ = evaluate(model, testloader, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        torch.save(checkpoint, os.path.join(checkpoint_dir, "last_checkpoint.pth"))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_experts = 16
    moe_config = ViTConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        num_hidden_layers=12,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        moe_shared_expert_intermediate_size=None,
        moe_shared_expert_overlap=False,
        recompute_granularity='none',
        recompute_modules=False,
        sequence_parallel=False,
        num_moe_experts=num_experts,
        calculate_per_token_loss=False,
        perform_initialization=False,
        moe_router_topk=1,
        moe_router_load_balancing_type='aux_loss',
        moe_router_score_function='softmax',
        moe_router_enable_expert_bias=False,
        _attn_implementation='eager',
        moe_input_jitter_eps=None, # float
        moe_router_dtype='fp32',
        moe_router_force_load_balancing=False,
        moe_z_loss_coeff=None, # 1e-3
        moe_router_pre_softmax=False, # softmax before topk(router)
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_router_topk_scaling_factor=None,
        moe_expert_capacity_factor=None,
        moe_aux_loss_coeff=1e-2,
        num_layers=0,
        mtp_num_layers=None,
        moe_ffn_hidden_size=3072, # hidden_size * 4
        gated_linear_unit=False,
        add_bias_linear=True,
        activation_func=F.gelu,
        moe_apply_probs_on_input=False,
        moe_permute_fusion=False,
    )
    layers_to_upcycle = [8, 9, 10, 11]  

    # ---------------- Stage 1: Dense ViT ----------------
    pretrained_model_name = "google/vit-base-patch16-224-in21k"
    dense_vit = AutoModelForImageClassification.from_pretrained(
        pretrained_model_name,
        num_labels=100
    )
    dense_vit.to(device)

    trainloader, testloader = get_cifar100_dataloaders(batch_size=64, data_dir="./data")

    print("=== Stage 1: Training Dense ViT ===")
    train_and_evaluate(trainloader, testloader, dense_vit, num_epochs=14, lr=5e-4, checkpoint_dir="./checkpoints_dense", device=device)

    # ---------------- Stage 2: Upcycling with MoE ----------------
    upcycled_model = VisionTransformerWithMoE(
        vit_model=dense_vit,       # Pass pretrained dense ViT
        num_experts=num_experts,
        moe_config=moe_config,
        moe_layer_indices=layers_to_upcycle
    )
    upcycled_model.to(device)

    print("=== Stage 2: Training Upcycled ViT with MoE ===")
    train_and_evaluate(trainloader, testloader, upcycled_model, num_epochs=14, lr=5e-4, checkpoint_dir="./checkpoints_moe", device=device)
