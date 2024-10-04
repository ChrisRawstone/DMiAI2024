from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
from torchvision import transforms, models
import numpy as np

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Define the Dataset class
class CTInpaintingDiffusionDataset2(Dataset):
    def __init__(self, data_dir, transform=None):
        self.corrupted_dir = os.path.join(data_dir, 'corrupted')
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.tissue_dir = os.path.join(data_dir, 'tissue')
        self.vertebrae_dir = os.path.join(data_dir, 'vertebrae')
        self.ct_dir = os.path.join(data_dir, 'ct')
        self.filenames = sorted(os.listdir(self.corrupted_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Load images
        corrupted = Image.open(os.path.join(self.corrupted_dir, filename)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, filename.replace('corrupted_', 'mask_'))).convert('L')
        tissue = Image.open(os.path.join(self.tissue_dir, filename.replace('corrupted_', 'tissue_'))).convert('L')
        ct = Image.open(os.path.join(self.ct_dir, filename.replace('corrupted_', 'ct_'))).convert('RGB')
        with open(os.path.join(self.vertebrae_dir, filename.replace('corrupted_', 'vertebrae_').replace('.png', '.txt')), 'r') as f:
            vertebrae_num = int(f.read().strip())
        
        
        # Transform images
        corrupted = self.transform(corrupted)
        mask = self.transform(mask)
        tissue = self.transform(tissue)
        ct = self.transform(ct)
    
        # Get the overlapping part of the mask
        temp_mask = tissue > 0
        overlapping_corruption_mask = mask * temp_mask

        
        return corrupted, overlapping_corruption_mask, ct, tissue, vertebrae_num

# Define the Dataset class
class CTInpaintingDiffusionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.corrupted_dir = os.path.join(data_dir, 'corrupted')
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.tissue_dir = os.path.join(data_dir, 'tissue')
        self.vertebrae_dir = os.path.join(data_dir, 'vertebrae')
        self.ct_dir = os.path.join(data_dir, 'ct')
        self.filenames = sorted(os.listdir(self.corrupted_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        base_filename = filename[len('corrupted_'):-len('.png')]
        patient_id_str, slice_num_str = base_filename.split('_')
        
        # Load images
        corrupted = Image.open(os.path.join(self.corrupted_dir, filename)).convert('L')
        mask = Image.open(os.path.join(self.mask_dir, filename.replace('corrupted_', 'mask_'))).convert('L')
        tissue = Image.open(os.path.join(self.tissue_dir, filename.replace('corrupted_', 'tissue_'))).convert('L')
        ct = Image.open(os.path.join(self.ct_dir, filename.replace('corrupted_', 'ct_'))).convert('L')

        # Load vertebrae number and normalize
        with open(os.path.join(self.vertebrae_dir, filename.replace('corrupted_', 'vertebrae_').replace('.png', '.txt')), 'r') as f:
            vertebrae_num = int(f.read().strip())
        vertebrae_normalized = (vertebrae_num - 1) / 32  # Normalized assuming vertebrae 1-33
        
        if self.transform:
            corrupted = self.transform(corrupted)
            mask = self.transform(mask)
            tissue = self.transform(tissue)
            ct = self.transform(ct)
            vertebrae_tensor = torch.full((1, corrupted.shape[1], corrupted.shape[2]), vertebrae_normalized)
        else:
            raise ValueError("Transform is required")

        
        temp_mask = tissue > 0
        
        # Extract the values of mask where the corrupted image is not zero
        overlapping_corruption_mask = mask * temp_mask
        
        
        # Combine inputs
        original_image = torch.cat([corrupted, tissue, vertebrae_tensor], dim=0)
        return original_image, overlapping_corruption_mask, ct
      
      
# Create a test for the dataset
if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = CTInpaintingDiffusionDataset(data_dir='data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        original_image, overlapping_corruption_mask, ground_truth_ct = batch
        print(original_image.shape, overlapping_corruption_mask.shape, ground_truth_ct.shape)
        
        # Create a figure showing the ground truth CT, and overlapping corruption mask
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(original_image[0, 0, :, :], cmap='gray')
        plt.title("Corrupted Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(overlapping_corruption_mask[0, 0, :, :], cmap='gray')
        plt.title("Overlapping Corruption Mask")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(ground_truth_ct[0, 0, :, :], cmap='gray')
        plt.title("Ground Truth CT")
        plt.axis('off')
        plt.savefig("test.png")   
        break