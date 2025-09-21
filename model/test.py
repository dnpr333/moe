import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import ml_collections
from vmoe import VisionTransformerMoe  
# from utils.load_pretrained import load_from_vit

def get_config(num_classes=10, variant_idx=0, num_total_experts=4,
               moe_where="Every", moe_where_num=2,
               num_selected_experts=2, group_size=1, patch_size=4):
    config = ml_collections.ConfigDict()
    config.name = 'VisionTransformerMoe'
    config.num_classes = num_classes
    config.patch_size = (patch_size, patch_size)
    config.hidden_size = [512, 768, 1024, 1280][variant_idx]
    config.classifier = 'token'
    config.representation_size = config.hidden_size
    config.head_bias_init = -math.log(num_classes)

    config.encoder = ml_collections.ConfigDict()
    config.encoder.hidden_size = config.hidden_size
    config.encoder.num_layers = [8, 12, 24, 32][variant_idx]
    config.encoder.mlp_dim = [2048, 3072, 4096, 5120][variant_idx]
    config.encoder.num_heads = [8, 12, 16, 16][variant_idx]
    config.encoder.dropout_rate = 0.0
    config.encoder.attention_dropout_rate = 0.0

    # MoE config
    config.encoder.moe = ml_collections.ConfigDict()
    config.encoder.moe.num_experts = num_total_experts
    if moe_where == 'Every':
        config.encoder.moe.layers = tuple(
            range(moe_where_num - 1, config.encoder.num_layers, moe_where_num))
    elif moe_where == 'Last':
        config.encoder.moe.layers = tuple(
            range(1, config.encoder.num_layers, 2))[-moe_where_num:]
    else:
        raise ValueError(f'Unknown moe_where={moe_where}, moe_where_num={moe_where_num}')
    config.encoder.moe.dropout_rate = 0.0
    # config.encoder.moe.split_rngs = False
    config.encoder.moe.group_size = group_size
    config.encoder.moe.router = ml_collections.ConfigDict()
    config.encoder.moe.router.num_experts = num_total_experts
    config.encoder.moe.router.num_selected_experts = num_selected_experts
    config.encoder.moe.router.noise_std = 1.0
    config.encoder.moe.router.importance_loss_weight = 0.005
    config.encoder.moe.router.load_loss_weight = 0.005
    config.encoder.moe.router.gshard_loss_weight = 0.0   
    config.encoder.moe.router.dispatcher = {}            
    config.encoder.moe.router.deterministic = False
    config.encoder.moe.router.dtype = None

    config.name_mapping = {
    "encoder.layers.0.mlp.fc1.weight": "encoder.layers.0.moe_mlp.fc1.weight",
    "encoder.layers.0.mlp.fc1.bias": "encoder.layers.0.moe_mlp.fc1.bias",
    "encoder.layers.0.mlp.fc2.weight": "encoder.layers.0.moe_mlp.fc2.weight",
    "encoder.layers.0.mlp.fc2.bias": "encoder.layers.0.moe_mlp.fc2.bias",
    }

    return config

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100. * correct / total
    return acc

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    # --- CIFAR10 dataset ---
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # random crop with padding
        transforms.RandomHorizontalFlip(),      # random left-right flip
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # brightness, contrast, saturation, hue
        transforms.RandomRotation(15),          # small random rotation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),  # CIFAR-10 mean/std
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=1)

    # --- Model ---
    config = get_config(num_classes=10, variant_idx=0, num_total_experts=4,
                        moe_where="Every", moe_where_num=2,
                        num_selected_experts=2, group_size=1, patch_size=4)
    model = VisionTransformerMoe(
        num_classes=config.num_classes,
        patch_size=config.patch_size,
        hidden_size=config.hidden_size,
        encoder=config.encoder,
        classifier=config.classifier,
        representation_size=config.representation_size,
        deterministic=False,
        head_bias_init=config.head_bias_init,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # --- Training Loop ---
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, metrics = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate
        val_acc = evaluate(model, testloader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {running_loss/len(trainloader):.4f} "
              f"Val Acc: {val_acc:.2f}%")

    print("Training complete!")

if __name__ == "__main__":
    main()
