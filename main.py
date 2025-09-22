# --- Example Usage ---
from model.vmoe_base import *
import timm
from train import train_one_epoch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from evaluate import evaluate
from earlystopping import EarlyStopping
from model.vit_moe import ViTMOE, MoeFFN
from tqdm import tqdm # For a nice progress bar
from transformers import ViTForImageClassification
class HFImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        # item["img"] is already a PIL.Image.Image
        img = item["img"]
        label = item["fine_label"]
        if self.transform:
            img = self.transform(img)
        return img, label
def main_training_loop(model, train_loader, val_loader, optimizer, config, num_epochs, device):
    
    model.to(device)
    classification_criterion = nn.CrossEntropyLoss()
    torch.autograd.set_detect_anomaly(True)
    
    # Initialize Early Stopping
    early_stopper = EarlyStopping(patience=3, checkpoint_path='vmoe_best.pth')

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, classification_criterion, config, device
        )
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Evaluate on the validation set
        val_loss, val_acc = evaluate(
            model, val_loader, classification_criterion, device
        )
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Check for early stopping
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            break
            
    print("Training finished.")
    print(f"Best validation loss: {early_stopper.best_loss:.6f}")
if __name__ == '__main__':
    total_depth = 12
    moe_layer_indices = tuple(range(1, total_depth, 2))
    config = {
    "num_classes": 100,
    "moe_layers": moe_layer_indices, 
    "num_experts": 8,
    "top_k": 2,
    "expert_dropout" : 0.3
    }
    
    print("--- Standard V-MoE ---")
    my_model = ViTMOE(config)
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-5
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),               
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                            std=[0.2675, 0.2565, 0.2761])
    ])
    from datasets import load_dataset
    hf_dataset = load_dataset("uoft-cs/cifar100")

    from torchvision.datasets.vision import VisionDataset
    from PIL import Image



    train_dataset = HFImageDataset(hf_dataset["train"], transform=train_transform)
    val_dataset   = HFImageDataset(hf_dataset["test"],  transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=4)
    # print(f"DataLoaders created. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    optimizer = optim.AdamW(my_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main_training_loop(
        my_model,
        train_loader,
        val_loader,
        optimizer,
        config,
        NUM_EPOCHS,
        DEVICE
    )
