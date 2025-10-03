from datasets import load_dataset
import os
from tqdm import tqdm
from PIL import Image

# Directory to save images
save_dir = "./imagenet"
os.makedirs(save_dir, exist_ok=True)

split = "train"  # or "validation"
dataset = load_dataset("imagenet-1k", split=split)

print(f"Total images in {split}: {len(dataset)}")

# Wrap dataset with tqdm
for i, item in enumerate(tqdm(dataset, desc=f"Downloading {split}", total=len(dataset))):
    image = item["image"]
    label = item["label"]
    label_dir = os.path.join(save_dir, split, str(label))
    os.makedirs(label_dir, exist_ok=True)
    
    # Save image
    image_path = os.path.join(label_dir, f"{i}.JPEG")
    image.save(image_path)
