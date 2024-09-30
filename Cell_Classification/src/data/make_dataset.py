import os
import sys
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def setup_working_directory():
    """
    Change the current working directory to the parent-parent directory of the script's directory.
    """
    script_dir = Path(__file__).resolve()
    parent_parent_dir = script_dir.parent.parent.parent
    os.chdir(parent_parent_dir)
    print(f"Changed working directory to: {parent_parent_dir}")

def decode_image(encoded_img: str) -> np.ndarray:
    """
    Decode a base64 encoded image string to a NumPy array.

    Args:
        encoded_img (str): Base64 encoded image.

    Returns:
        np.ndarray: Decoded image as a NumPy array.
    """
    np_img = np.frombuffer(base64.b64decode(encoded_img), np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_ANYCOLOR)

def tif_to_ndarray(tif_path: str) -> np.ndarray:
    """
    Load a .tif image as a NumPy array.

    Args:
        tif_path (str): Path to the .tif image.

    Returns:
        np.ndarray: Image array.
    """
    img_array = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if img_array is None:
        raise ValueError(f"Failed to load image at path: {tif_path}")
    return img_array

def convert_16bit_to_8bit(image: np.ndarray) -> np.ndarray:
    """
    Convert a 16-bit image to 8-bit by normalizing the pixel values to the 0-255 range.

    Args:
        image (np.ndarray): 16-bit image.

    Returns:
        np.ndarray: Normalized 8-bit image.
    """
    if image.dtype != np.uint16:
        raise ValueError("Image is not 16-bit.")

    # Normalize the image to 0-255
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image_8bit = image_8bit.astype(np.uint8)
    return image_8bit

class LoadTifDataset(Dataset):
    def __init__(self, image_dir: str, csv_file_path: str, transform=None):
        """
        Custom Dataset for loading .tif images and their labels from a CSV file.

        Args:
            image_dir (str): Path to the folder containing .tif images.
            csv_file_path (str): Path to the CSV file with image file names and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(csv_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def load_data(self, idx: int):
        """
        Loads a single image and its label given an index.

        Args:
            idx (int): Index of the image in the CSV file.

        Returns:
            tuple: (image, label) where image is the loaded and optionally transformed image and
                   label is a tensor representing the image's label.
        """
        # Get the image file name and label from the CSV file
        img_name = str(self.labels_df.iloc[idx, 0]).zfill(3)
        label = self.labels_df.iloc[idx, 1]

        # Full path to the .tif image
        img_path = os.path.join(self.image_dir, f"{img_name}.tif")

        # Load the .tif image using tif_to_ndarray
        image = tif_to_ndarray(img_path)

        # Convert 16-bit to 8-bit
        image = convert_16bit_to_8bit(image)

        # Convert ndarray to PIL Image for transformations
        image = Image.fromarray(image)

        # Convert grayscale to RGB by replicating channels
        image = image.convert('RGB')

        # Apply the transformations
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx: int):
        """
        PyTorch will call this method to get a single image-label pair during data loading.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            tuple: (image, label) as returned by load_data function.
        """
        return self.load_data(idx)

# def save_images_as_grid(dataloader, classes, save_path, dataset_name):
#     """
#     Save all images from the dataloader in a single grid image.

#     Args:
#         dataloader (DataLoader): DataLoader for the dataset.
#         classes (list): List of class names.
#         save_path (str): Path where the grid image will be saved.
#         dataset_name (str): Name of the dataset ('train' or 'val') for labeling.
#     """
#     images_list = []
#     labels_list = []

#     # Collect all images and labels from the dataloader
#     for images, labels in dataloader:
#         images_list.append(images)
#         labels_list.append(labels)

#     # Concatenate all batches
#     images = torch.cat(images_list, dim=0)
#     labels = torch.cat(labels_list, dim=0)

#     # Limit the number of images if necessary
#     max_images = 64  # Adjust this number based on how many images you want to display
#     if images.size(0) > max_images:
#         images = images[:max_images]
#         labels = labels[:max_images]

#     # Number of images
#     num_images = images.size(0)
#     # Calculate grid size
#     grid_cols = 8  # Adjust as needed
#     grid_rows = (num_images + grid_cols - 1) // grid_cols

#     # Create subplots
#     fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))

#     # Flatten axes for easy iteration
#     axes = axes.flatten()

#     for idx in range(num_images):
#         image = images[idx]
#         label = labels[idx].item()
#         class_name = classes[label] if classes else str(label)

#         # Convert tensor to numpy array
#         img_np = image.permute(1, 2, 0).numpy()
#         img_np = np.clip(img_np, 0, 1)  # Ensure the image is in [0,1]

#         # Display the image
#         axes[idx].imshow(img_np)
#         axes[idx].set_title(f"{class_name}")
#         axes[idx].axis('off')

#     # Hide any unused subplots
#     for idx in range(num_images, len(axes)):
#         axes[idx].axis('off')

#     plt.tight_layout()
#     # Save the figure
#     grid_filename = os.path.join(save_path, f"{dataset_name}_grid.png")
#     plt.savefig(grid_filename, dpi=300)
#     plt.close(fig)
#     print(f"Saved {dataset_name} image grid to {grid_filename}")

# def main():
#     # No need to change working directory in this example
#     setup_working_directory()

#     # Define transformations for training data
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor(),
#     ])

#     # Define transformations for validation data (no augmentation)
#     val_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor(),
#     ])

#     image_dir = "data/training"
#     csv_file_path = "data/training.csv"

#     image_dir_val = "data/validation16bit"
#     csv_file_path_val = "data/validation.csv"

#     # Create the dataset with training transformations
#     dataset_train = LoadTifDataset(image_dir=image_dir, csv_file_path=csv_file_path, transform=train_transform)
#     dataset_val = LoadTifDataset(image_dir=image_dir_val, csv_file_path=csv_file_path_val, transform=val_transform)

#     # Create the dataloader for batch processing
#     dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=False)
#     dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False)

    # # Example class names, modify as needed
    # classes = ['class0', 'class1']  # Replace with actual class names

    # # Directory where images will be saved
    # save_directory = "saved_images"
    # os.makedirs(save_directory, exist_ok=True)

    # # Save training images grid
    # save_images_as_grid(dataloader_train, classes, save_directory, dataset_name='train')

    # # Save validation images grid
    # save_images_as_grid(dataloader_val, classes, save_directory, dataset_name='val')

# if __name__ == "__main__":
#     main()
