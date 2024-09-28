import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from pathlib import Path
import os
import sys
import pandas as pd


from src.utils import tif_to_ndarray

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
        """
        Make PadToSize callable so it can be used inside transforms.Compose.
        
        Args:
            img (PIL.Image): The image to pad.

        Returns:
            PIL.Image: Padded image.
        """
        width, height = img.size
        
        # Calculate padding required on each side
        pad_width = max(0, (self.target_width - width) // 2)
        pad_height = max(0, (self.target_height - height) // 2)
        
        # Apply padding to the image
        padding = (pad_width, pad_height, self.target_width - width - pad_width, self.target_height - height - pad_height)
        padded_img = ImageOps.expand(img, padding)
        
        return padded_img

class LoadTifDataset(Dataset):
    def __init__(self, image_dir, csv_file_path, transform=None):
        """
        Custom Dataset for loading .tif images and their labels from a CSV file.

        Args:
            image_dir (str): Path to the folder containing .tif images.
            csv_file_path (str): Path to the CSV file with image file names and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        #script_dir = os.path.dirname(os.path.realpath(__file__))
        self.labels_df = pd.read_csv(csv_file_path)
        self.transform = transform

    def __len__(self):
        # Returns the total number of samples (rows) in the dataset
        return len(self.labels_df)

    def load_data(self, idx):
        """
        Loads a single image and its label given an index.

        Args:
            idx (int): Index of the image in the CSV file.

        Returns:
            tuple: (image, label) where image is the loaded and optionally transformed image and
                label is a tensor representing the image's label.
        """
        # Get the image file name and label from the CSV file
        img_name = str(self.labels_df.iloc[idx, 0]).zfill(3)  # Zero-pad to 3 digits
        label = self.labels_df.iloc[idx, 1]                   # Second column: label (0 or 1)
        
        # Full path to the .tif image
        img_path = os.path.join(self.image_dir, f"{img_name}.tif")
        
        # Load the .tif image using tif_to_ndarray
        image = tif_to_ndarray(img_path)
        
        # Check if the image was loaded correctly
        if image is None:
            raise ValueError(f"Failed to load image at path: {img_path}")
        
        # Convert ndarray to PIL Image for transformations
        image = Image.fromarray(image)
        
        # Apply the transformations, including padding
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx):
        """
        PyTorch will call this method to get a single image-label pair during data loading.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            tuple: (image, label) as returned by load_data function.
        """
        return self.load_data(idx)

# Define the padding size
target_width = 1500
target_height = 1470

# Define transformations
transform = transforms.Compose([
    PadToSize(target_width=target_width, target_height=target_height),  # Custom padding transform
    transforms.ToTensor()])

# Example usage
image_dir = "data/training"
csv_file_path = "data/training.csv"

# Create the dataset with padding transformation
dataset = LoadTifDataset(image_dir=image_dir, csv_file_path=csv_file_path, transform=transform)

# Create the dataloader for batch processing
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)