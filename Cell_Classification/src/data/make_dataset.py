
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps
from pathlib import Path
import os
import pandas as pd
import numpy as np
import random

# Get the script's directory
script_dir = Path(__file__).resolve()
# Go two levels up from the script's directory
parent_parent_dir = script_dir.parent.parent.parent
# Change the current working directory to the parent-parent directory
os.chdir(parent_parent_dir)

# Define PadToSize to work with transforms.Compose
class PadToSize:
    def __init__(self, target_width, target_height):
        self.target_width = target_width
        self.target_height = target_height

    def __call__(self, img):
        # Get the current dimensions of the image
        width, height = img.size

        # Calculate padding on each side to center the image
        pad_width = max(0, (self.target_width - width) // 2)
        pad_height = max(0, (self.target_height - height) // 2)

        # Apply padding
        padding = (pad_width, pad_height, self.target_width - width - pad_width, self.target_height - height - pad_height)
        return ImageOps.expand(img, padding, fill=0) # Black padding

class LoadTifDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = str(self.labels_df.iloc[idx, 0]).zfill(3)  # Zero-pad to 3 digits
        label = self.labels_df.iloc[idx, 1]  # Get the label (0 or 1)
        
        # Full path to the .tif image
        img_path = os.path.join(self.image_dir, f"{img_name}.tif")
        
        # Open the image and convert to grayscale mode "L"
        image = Image.open(img_path)     #.convert('L')  # Ensures image is 8-bit grayscale
        
        # If image is 16-bit, convert it to 8-bit grayscale
        if image.mode == 'I;16B':
            # Convert 16-bit to 8-bit by scaling down
            image = (np.array(image) / 256).astype(np.uint8)  # Scale pixel values to 8-bit
            image = Image.fromarray(image)  # Convert back to PIL Image
            image = image.convert('L')  # Ensure it's in 8-bit grayscale

        # Ensure all images are in 8-bit grayscale mode
        # else:
        #     image = image.convert('L')

        image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


final_resize_size = (224, 224)
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
    #PadToSize(*target_size),  # Pad to the target size of the largest image
    transforms.Resize(final_resize_size),  # Resize the shorter side first
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
    #PadToSize(*target_size),  # Pad to the target size of the largest image
    transforms.Resize(final_resize_size),  # Resize the shorter side first
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize as per ImageNet
])

# Paths to the CSV file and image directory
csv_file = 'data/training.csv'
image_dir = 'data/training/'

csv_file_val = 'data/validation.csv'
image_dir_val = 'data/validation/'

csv_file_aug = 'data/augmented_training.csv'
image_dir_aug = 'data/train_augmentation/'

# Create datasets
train_dataset = LoadTifDataset(csv_file=csv_file, image_dir=image_dir, transform=train_transform)
train_aug_dataset = LoadTifDataset(csv_file=csv_file_aug, image_dir=image_dir_aug, transform=train_transform)
val_dataset = LoadTifDataset(csv_file=csv_file_val, image_dir=image_dir_val, transform=val_transform)


# Combine the train_dataset and train_aug_dataset
combined_train_dataset = ConcatDataset([train_dataset, train_aug_dataset])

# Create DataLoader objects
train_dataloader = DataLoader(combined_train_dataset, batch_size=64, shuffle=True, num_workers=4)
# train_dataloader_aug = DataLoader(train_aug_dataset, batch_size=64, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

print("Data loaded successfully")
