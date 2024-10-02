# make_dataset.py
import os
import base64
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import logging

def setup_working_directory():
    """
    Change the current working directory to the parent-parent directory of the script's directory.
    """
    script_dir = Path(__file__).resolve()
    parent_parent_dir = script_dir.parent.parent.parent
    os.chdir(parent_parent_dir)
    print(f"Changed working directory to: {parent_parent_dir}")

setup_working_directory()

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
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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

        # Convert grayscale to RGB by replicating channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply the transformations
        
        image = image.astype(np.float32) / 255.0
        
        if self.transform:
            # Albumentations expects numpy array and returns a dictionary
            transformed = self.transform(image=image)
            image = transformed['image']

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


def get_transforms(img_size: int) -> Tuple[A.Compose, A.Compose]:
    """
    Define data augmentation and preprocessing transforms for training and validation.

    Args:
        img_size (int): Image size for resizing.

    Returns:
        Tuple[A.Compose, A.Compose]: Training and validation transforms.
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Normalize(mean=(0.485, 0.485, 0.485),  
                    std=(0.229, 0.229, 0.229)),  
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.485, 0.485), 
                    std=(0.229, 0.229, 0.229)), 
        ToTensorV2(),
    ])

    logging.info("Transforms for training and validation have been defined.")
    return train_transform, val_transform


def create_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle class imbalance.

    Args:
        labels (np.ndarray): Array of labels.

    Returns:
        WeightedRandomSampler: Sampler object.
    """
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    if num_classes < 2:
        raise ValueError("Number of classes should be at least 2 for WeightedRandomSampler.")

    # Compute weights for each class
    weights_per_class = total_samples / (num_classes * class_counts)
    weights = weights_per_class[labels]

    sampler = WeightedRandomSampler(weights, num_samples=total_samples, replacement=True)
    logging.info("WeightedRandomSampler has been created.")
    return sampler


def get_dataloaders(batch_size: int, img_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoader objects for training and validation datasets.

    Args:
        batch_size (int): Batch size for DataLoaders.
        img_size (int): Image size for resizing.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    # Define paths
    train_image_dir = Path("data/training")
    train_csv_path = Path("data/training.csv")
    val_image_dir = Path("data/validation")
    val_csv_path = Path("data/validation.csv")

    # Get transforms
    train_transform, val_transform = get_transforms(img_size)

    # Create the datasets
    train_dataset = LoadTifDataset(
        image_dir=train_image_dir,
        csv_file_path=train_csv_path,
        transform=train_transform
    )
    val_dataset = LoadTifDataset(
        image_dir=val_image_dir,
        csv_file_path=val_csv_path,
        transform=val_transform
    )

    logging.info("Datasets for training and validation have been created.")

    # Extract labels from the training dataset
    train_labels = train_dataset.labels_df.iloc[:, 1].values  # assuming labels are in the second column

    # Create sampler for training
    sampler = create_sampler(train_labels)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    logging.info("DataLoaders for training and validation have been initialized.")
    return train_loader, val_loader


def get_dataloaders_final_train(batch_size: int, img_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoader objects for training and validation datasets.

    Args:
        batch_size (int): Batch size for DataLoaders.
        img_size (int): Image size for resizing.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    # Define paths
    # train_image_dir = Path("data/training_val")
    # train_csv_path = Path("data/training_val.csv")
    
    # val_image_dir = Path("data/test_val")
    # val_csv_path = Path("data/test_val.csv")
    
    train_image_dir = Path("data/training")
    train_csv_path = Path("data/training.csv")
    
    val_image_dir = Path("data/validation")
    val_csv_path = Path("data/validation.csv")

    # Get transforms
    train_transform, val_transform = get_transforms(img_size)

    # Create the datasets
    train_dataset = LoadTifDataset(
        image_dir=train_image_dir,
        csv_file_path=train_csv_path,
        transform=train_transform
    )
    
    train_dataset_eval = LoadTifDataset(
        image_dir=train_image_dir,
        csv_file_path=train_csv_path,
        transform=val_transform
    )
    
    val_dataset = LoadTifDataset(
        image_dir=val_image_dir,
        csv_file_path=val_csv_path,
        transform=val_transform
    )

    logging.info("Datasets for training and validation have been created.")

    # Extract labels from the training dataset
    train_labels = train_dataset.labels_df.iloc[:, 1].values  # assuming labels are in the second column

    # Create sampler for training
    sampler = create_sampler(train_labels)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True
    )
    
    train_loader_eval = DataLoader(
        train_dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    logging.info("DataLoaders for training and validation have been initialized.")
    return train_loader, val_loader, train_loader_eval


def get_dataloaders_train(batch_size: int, img_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoader objects for training and validation datasets by splitting the training data.

    Args:
        batch_size (int): Batch size for DataLoaders.
        img_size (int): Image size for resizing.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    # Define paths
    train_image_dir = Path("data/training_val")
    train_csv_path = Path("data/training_val.csv")

    # Get transforms
    train_transform, val_transform = get_transforms(img_size)

    # Create the full dataset
    full_dataset = LoadTifDataset(
        image_dir=train_image_dir,
        csv_file_path=train_csv_path,
        transform=None  # Transforms will be set later
    )

    # Extract labels from the full dataset
    labels = full_dataset.labels_df.iloc[:, 1].values  # Assuming labels are in the second column

    # Get indices
    indices = np.arange(len(full_dataset))

    # Stratified splitting
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.15,
        stratify=labels,
        random_state=42
    )

    # Subset the datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # Set transforms for each subset
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Extract labels for the training dataset (after splitting)
    train_labels = labels[train_indices]

    # Create sampler for training
    sampler = create_sampler(train_labels)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return train_loader, val_loader

def save_images_as_grid(dataloader, classes, save_path, dataset_name):
    """
    Save all images from the dataloader in a single grid image.

    Args:
        dataloader (DataLoader): DataLoader for the dataset.
        classes (list): List of class names.
        save_path (str): Path where the grid image will be saved.
        dataset_name (str): Name of the dataset ('train' or 'val') for labeling.
    """
    images_list = []
    labels_list = []

    # Collect all images and labels from the dataloader
    for images, labels in dataloader:
        images_list.append(images)
        labels_list.append(labels)

    # Concatenate all batches
    images = torch.cat(images_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # Limit the number of images if necessary
    max_images = 64  # Adjust this number based on how many images you want to display
    if images.size(0) > max_images:
        images = images[:max_images]
        labels = labels[:max_images]

    # Number of images
    num_images = images.size(0)
    # Calculate grid size
    grid_cols = 8  # Adjust as needed
    grid_rows = (num_images + grid_cols - 1) // grid_cols

    # Define mean and std used in A.Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Create subplots
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for idx in range(num_images):
        image = images[idx]
        label = labels[idx].item()
        class_name = classes[label] if classes else str(label)

        # Unnormalize the image
        img_np = image.permute(1, 2, 0).numpy()
        img_np = std * img_np + mean  # Unnormalize
        img_np = np.clip(img_np, 0, 1)  # Ensure the image is in [0,1]

        # Display the image
        axes[idx].imshow(img_np)
        axes[idx].set_title(f"{class_name}")
        axes[idx].axis('off')

    # Hide any unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    # Save the figure
    grid_filename = os.path.join(save_path, f"{dataset_name}_grid.png")
    plt.savefig(grid_filename, dpi=300)
    plt.close(fig)
    print(f"Saved {dataset_name} image grid to {grid_filename}")

def main():
    # No need to change working directory in this example
    setup_working_directory()

    # Define transformations for training data
    img_size = 224
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),  # Using ImageNet means
                    std=(0.229, 0.224, 0.225)),   # Using ImageNet stds
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),  # Using ImageNet means
                    std=(0.229, 0.224, 0.225)),   # Using ImageNet stds
        ToTensorV2(),
    ])

    image_dir = "data/training"
    csv_file_path = "data/training.csv"

    image_dir_val = "data/validation16bit"
    csv_file_path_val = "data/validation.csv"

    # Create the dataset with training transformations
    dataset_train = LoadTifDataset(image_dir=image_dir, csv_file_path=csv_file_path, transform=train_transform)
    dataset_val = LoadTifDataset(image_dir=image_dir_val, csv_file_path=csv_file_path_val, transform=val_transform)

    # Create the dataloader for batch processing
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=False)
    dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False)

    # Example class names, modify as needed
    classes = ['class0', 'class1']  # Replace with actual class names

    # Directory where images will be saved
    save_directory = "saved_images"
    os.makedirs(save_directory, exist_ok=True)

    # Save training images grid
    save_images_as_grid(dataloader_train, classes, save_directory, dataset_name='train')

    # Save validation images grid
    save_images_as_grid(dataloader_val, classes, save_directory, dataset_name='val')

if __name__ == "__main__":
    main()
