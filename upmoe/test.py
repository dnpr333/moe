import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import ViTConfig, AutoImageProcessor, AutoModelForImageClassification
from dataloader import get_cifar100_dataloaders
from model import VisionTransformerWithMoE
from utils import EarlyStopping

import os

def evaluate(model, testloader, device, aux_loss_weight=0.01, k=1):
    model.eval()

    total_samples = 0
    total_correct = 0
    total_loss, total_cls_loss, total_aux_loss = 0.0, 0.0, 0.0
    correct_at_k = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in testloader:
            labels = labels.to(device)
            imgs = imgs.to(device)

            logits, aux_loss = model(imgs)

            cls_loss = criterion(logits, labels)
            aux_loss = aux_loss if aux_loss is not None else torch.tensor(0.0, device=device)
            loss = cls_loss + aux_loss_weight * aux_loss

            # accuracy
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            total_samples += labels.size(0)
            total_correct += (preds == labels).sum().item()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_aux_loss += aux_loss.item()

            # precision @1
            topk_preds = logits.topk(k, dim=1).indices
            correct_at_k += sum([labels[i] in topk_preds[i] for i in range(labels.size(0))])

    accuracy = 100.0 * total_correct / total_samples
    precision_at_k = 100.0 * correct_at_k / total_samples
    avg_loss = total_loss / len(testloader)

    print(
        f"Eval | Loss: {avg_loss:.4f} "
        f"(CLS: {total_cls_loss/len(testloader):.4f}, AUX: {total_aux_loss/len(testloader):.4f}) "
        f"| Accuracy: {accuracy:.2f}% | Precision@{k}: {precision_at_k:.2f}%"
    )

    return accuracy, precision_at_k, avg_loss

# def train_and_evaluate(
#     trainloader,
#     testloader,
#     model,
#     num_epochs=50,
#     lr=1e-3,
#     checkpoint_dir="./checkpoints",
#     resume=False,
#     patience=5
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     optimizer = optim.AdamW(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
#     early_stopper = EarlyStopping(patience=patience)
#     start_epoch = 0

#     if resume:
#         checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
#         if os.path.exists(checkpoint_path):
#             print(f"Resuming training from {checkpoint_path}")
#             checkpoint = torch.load(checkpoint_path, map_location=device)
#             model.load_state_dict(checkpoint["model_state"])
#             optimizer.load_state_dict(checkpoint["optimizer_state"])
#             start_epoch = checkpoint["epoch"] + 1

#     os.makedirs(checkpoint_dir, exist_ok=True)

#     for run_idx, loader in enumerate(trainloader):
#         print(f"=== Run {run_idx+1} ===")
#         for epoch in range(start_epoch, num_epochs):
#             model.train()
            
#             # # warm up step for cls
#             # if epoch < 1:
#             #     for name, param in model.named_parameters():
#             #         if "classifier" not in name:
#             #             param.requires_grad = False
#             # else:
#             #     for param in model.parameters():
#             #         param.requires_grad = True

#             total_loss, total_cls_loss, total_aux_loss = 0.0, 0.0, 0.0
#             total_samples, total_correct, correct_at_k = 0, 0, 0
            
#             for imgs, labels in loader:
#                 labels = labels.to(device)
#                 imgs = imgs.to(device)

#                 logits, aux_loss = model(imgs)

#                 cls_loss = criterion(logits, labels)
#                 aux_loss = aux_loss if aux_loss is not None else torch.tensor(0.0, device=device)
#                 loss = cls_loss + 0.01 * aux_loss

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 total_loss += loss.item()
#                 total_cls_loss += cls_loss.item()
#                 total_aux_loss += aux_loss.item() 

#                 # Accuracy
#                 pred_probs = F.softmax(logits, dim=1)
#                 preds = pred_probs.argmax(dim=1)
#                 total_samples += labels.size(0)
#                 total_correct += (preds == labels).sum().item()

#                 # Precision@1
#                 topk_preds = logits.topk(1, dim=1).indices
#                 correct_at_k += sum([labels[i] in topk_preds[i] for i in range(labels.size(0))])

#             accuracy = 100.0 * total_correct / total_samples
#             precision_at_1 = 100.0 * correct_at_k / total_samples

#             print(
#                 f"Run {run_idx} - Epoch {epoch+1} | Train Loss: {total_loss/len(loader):.4f} "
#                 f"(CLS: {total_cls_loss/len(loader):.4f}, AUX: {total_aux_loss/len(loader):.4f}) "
#                 f"| Acc: {accuracy:.2f}% | Precision@1: {precision_at_1:.2f}%"
#             )

#             accuracy, precision_at_1, avg_loss = evaluate(model, testloader, device)
#             print(f"Run {run_idx} - Val acc={accuracy:.4f}, Val P@1={precision_at_1:.4f}, val avg loss={avg_loss:.4f}")

#             checkpoint = {
#                 "epoch": epoch,
#                 "model_state": model.state_dict(),
#                 "optimizer_state": optimizer.state_dict(),
#             }
#             torch.save(checkpoint, os.path.join(checkpoint_dir, "last_checkpoint.pth"))
#             torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

#             # if early_stopper.step(accuracy):
#             #     print(f"Early stopping at epoch {epoch+1}")
#             #     break

#         print(f"Training complete for {run_idx+1}")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

class EarlyStopping:
    """Simple early stopping utility."""
    def __init__(self, patience=5, delta=0.0, save_path=None):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.save_path is not None:
                torch.save(model.state_dict(), self.save_path)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path is not None:
                torch.save(model.state_dict(), self.save_path)

def train_and_evaluate(
    trainloader,
    testloader,
    model,
    num_epochs=50,
    lr=1e-3,
    checkpoint_dir="./checkpoints",
    patience=5,
    run_idx=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=patience, save_path=os.path.join(checkpoint_dir, "best_model.pth"))

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_cls_loss, total_aux_loss = 0.0, 0.0, 0.0
        total_samples, total_correct, correct_at_k = 0, 0, 0

        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)

            # dense model
            if isinstance(outputs, tuple) and len(outputs) == 2:
                logits, aux_loss = outputs
            else:
                logits, aux_loss = outputs, None

            cls_loss = criterion(logits, labels)
            aux_loss = aux_loss if aux_loss is not None else torch.tensor(0.0, device=device)
            loss = cls_loss + 0.01 * aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_aux_loss += aux_loss.item() if aux_loss is not None else 0.0

            # Accuracy
            preds = logits.argmax(dim=1)
            total_samples += labels.size(0)
            total_correct += (preds == labels).sum().item()
            topk_preds = logits.topk(1, dim=1).indices
            correct_at_k += sum([labels[i] in topk_preds[i] for i in range(labels.size(0))])

        scheduler.step() 

        train_acc = 100.0 * total_correct / total_samples
        train_prec1 = 100.0 * correct_at_k / total_samples
        print(f"Run {run_idx} - Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {total_loss/len(trainloader):.4f} "
              f"(CLS: {total_cls_loss/len(trainloader):.4f}, AUX: {total_aux_loss/len(trainloader):.4f}) | "
              f"Acc: {train_acc:.2f}% | Precision@1: {train_prec1:.2f}%")

        # val
        val_acc, val_prec1, val_loss = evaluate(model, testloader, device)
        print(f"Run {run_idx} - Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Precision@1: {val_prec1:.2f}%")

        early_stopper.step(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, "last_checkpoint.pth"))
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = "cpu"

    print(f"Using device: {device}")

    num_experts = 32
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

    dataset = "cifar100"
    if dataset == "cifar100":
        trainloader, testloader = get_cifar100_dataloaders(batch_size=64, 
                                                           data_dir='./data', 
                                                           num_workers=0, 
                                                        #    num_shot=1, 
                                                        #    k_shot=10, 
                                                           seed=42)
        moe_type = "last_2"

    num_layers = 12

    if moe_type == "last_6":
        layers_to_use = [i for i in range(num_layers - 2, num_layers)]  
    elif moe_type == "even":
        layers_to_use = [i for i in range(num_layers) if i % 2 == 0]  
    elif moe_type == "last_2":
        layers_to_use = [num_layers-1, num_layers-3]  # every last 2
    else:
        raise ValueError(f"Invalid moe_type: {moe_type}. Choose 'last_6' or 'even'/'last 2.")
    
    model = VisionTransformerWithMoE(
        pretrained_model_name="google/vit-base-patch16-224-in21k",
        num_experts=num_experts,
        moe_config=moe_config,
        moe_layer_indices=layers_to_use
    )  

    print(model)
    model.to(device)

    train_and_evaluate(
        trainloader,
        testloader,
        model,
        num_epochs=100,
        lr=1e-4,
        checkpoint_dir="./checkpoints",
        patience=5
    )