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
vit_s_pretrained = timm.create_model('vit_small_patch32_224.augreg_in21k', pretrained=True)
pretrained_dict = vit_s_pretrained.state_dict()
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
    
    # Initialize Early Stopping
    early_stopper = EarlyStopping(patience=5, checkpoint_path='vmoe_best.pth')

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
    # You can now load the best model for inference
    # model.load_state_dict(torch.load('vmoe_best.pth'))
if __name__ == '__main__':
    total_depth = 12
    # Programmatically create a tuple of even indices: (0, 2, 4)
    moe_layer_indices = tuple(range(0, total_depth, 2))
    # 1. Define a configuration dictionary
    config = {
        'image_size': 224,
        'patch_size': 32,
        'num_classes': 100,
        'dim': 384,
        'depth': total_depth,
        'num_heads': 6,
        'mlp_hidden_dim': 1536,
        'encoder_config': {
            'moe': {
                'layers': moe_layer_indices,  
                'num_experts': 8,
                'mlp_dim': 384,
                'router': {
                    'num_selected_experts': 2,
                    'noise_std': 1/8
                }
            }
        }
    }
    
    # 2. Create the V-MoE model
    print("--- Standard V-MoE ---")
    my_model = VisionTransformerMoe(**config)
    my_model_dict = my_model.state_dict()
    new_state_dict = {}
    for key, value in pretrained_dict.items():
        if key in my_model_dict:
            # Kiểm tra xem kích thước tensor có khớp không
            if value.shape == my_model_dict[key].shape:
                new_state_dict[key] = value
                print(key)
            else:
                print(f"Skipping {key}: shape mismatch. Pretrained: {value.shape}, Model: {my_model_dict[key].shape}")
        else:
            # Một số key có thể cần đổi tên, ví dụ từ mlp -> moe.experts
            # Đây là một ví dụ đơn giản, bạn cần điều chỉnh cho phù hợp
            # Ví dụ này chỉ nạp các phần chung, bỏ qua các lớp mlp/moe
            pass

    print(f"Loaded {len(new_state_dict)} matching layers.")
    my_model.load_state_dict(new_state_dict, strict=False)
    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 1e-4
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config['image_size']),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),                # short side to 256
        transforms.CenterCrop(config['image_size']),
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, 
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print(f"DataLoaders created. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
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
