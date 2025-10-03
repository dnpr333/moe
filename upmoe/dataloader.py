from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch

def get_fewshot_indices(dataset, k_shot, seed=None):
    """
    Return indices for k-shot sampling per class.
    """
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    indices = []

    rng = np.random.default_rng(seed)
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        chosen = rng.choice(cls_indices, size=k_shot, replace=False)
        indices.extend(chosen.tolist())

    return indices


def get_cifar100_dataloaders(batch_size=64, data_dir='./data', num_workers=1, 
                             num_shot=1, k_shot=10, seed=42):
    """
    num_shot: how many different few-shot runs (episodes) to prepare
    k_shot: number of samples per class (e.g., 10 for 10-shot learning)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        transforms.RandomRotation(15),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1,0.1))], p=0.5),  # slight translation
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Load datasets
    full_trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    print(f"Total number of training images: {len(full_trainset)}")
    print(f"Total number of testing images: {len(testset)}")
    # # Few-shot training loaders
    # trainloaders = []
    # for run in range(num_shot):
    #     indices = get_fewshot_indices(full_trainset, k_shot=k_shot, seed=seed+run)
    #     subset = Subset(full_trainset, indices)
    #     loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #     trainloaders.append(loader)

    # print(f"num trainloaders: {len(trainloaders)}")
    trainloader = DataLoader(
        full_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return trainloader, testloader