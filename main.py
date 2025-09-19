import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

from model.vmoe_base import VisionTransformerMoe

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load config 
    config = load_config()
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["dataset"]

    # Load pretrained ViT
    vit_s_pretrained = timm.create_model(
        "vit_small_patch32_224.augreg_in21k", pretrained=True
    )
    # vit_s_pretrained = timm.create_model("vit_base_patch16_224.orig_in21k", pretrained=True)
    pretrained_dict = vit_s_pretrained.state_dict()

    # init model
    model = VisionTransformerMoe(**model_cfg)
    model_dict = model.state_dict()
    # Match pretrained weights where possible
    new_state_dict = {}
    for key, value in pretrained_dict.items():
        if key in model_dict and value.shape == model_dict[key].shape:
            new_state_dict[key] = value
        else:
            print(f"Skipping {key}: mismatch or missing.")
    print(f"Loaded {len(new_state_dict)} matching layers.")
    model.load_state_dict(new_state_dict, strict=False)

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(data_cfg["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(data_cfg["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )
    val_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=val_transform,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    print(f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Training Loop
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device using: {device}")

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):  # small run
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch+1}, {i+1}] loss: {running_loss/100:.3f}")
                running_loss = 0.0

if __name__ == "__main__":
    main()
