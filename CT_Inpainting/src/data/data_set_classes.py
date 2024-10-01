import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import wandb  # Import wandb for Weights and Biases integration
from tqdm import tqdm  # Import tqdm for progress bars
import matplotlib.pyplot as plt  # Import matplotlib for visualization
import numpy as np
from src.models.model import UNet
from src.data.data_set_augmentations import flipMaskAug
import datetime
import shutil



#base class for both classifier and inpainting
class BaseClass(Dataset):
    def __init__(self, data_dir, transform=None, dirs= ['corrupted', 'mask', 'tissue', 'vertebrae', 'ct'], desired_input = ['corrupted', 'mask', 'tissue', 'vertebrae'], desired_output = ['ct'],crop_mask=False):
        if transform is None:
            assert False, "Transform is required"
        self.data_dir = data_dir
        self.dirs = dirs
        for dir in self.dirs:
            setattr(self, dir + '_dir', os.path.join(data_dir, dir))
        
        self.crop_mask = crop_mask

        self.desired_input = desired_input
        self.desired_output = desired_output

        # Get list of corrupted images
        self.identifers = self.get_stripped_filename(sorted(os.listdir(getattr(self, dirs[0] + '_dir'))))
        self.transform = transform
    
    def get_stripped_filename(self, filenames):
        # check if filenames are in the format corrupted_000_1.png or just corrupted_0.png
        if len(filenames[0].split('_')) == 3:

            return [filename.split('_')[1] + "_" + filename.split('_')[2].split('.')[0] for filename in filenames]     
        else:
            return [filename.split('_')[1].split('.')[0] for filename in filenames]

    def __len__(self):
        return len(self.identifers)
    
    def __getitem__(self, idx):
        # Get the filename
        filename = self.identifers[idx]  # e.g., '000_1'
        # Construct file paths
        path_to_files = {f"{dir}_path": os.path.join(getattr(self, dir + '_dir'), dir + '_' + filename + '.png') for dir in self.dirs}
        # change vertebrate path to .txt
        
        if 'vertebrae' in self.dirs:
            path_to_files['vertebrae_path'] = path_to_files['vertebrae_path'].replace('.png', '.txt')
       
        # Load images   
        images = {dir.split("_")[0]: Image.open(path).convert('L') for dir, path in path_to_files.items() if dir != 'vertebrae_path'}
        tensors = {key: self.transform(image) for key, image in images.items()}
        if 'vertebrae' in self.dirs:
            with open(path_to_files['vertebrae_path'], 'r') as f:
                vertebrae_num = int(f.read().strip())
            # Normalize the vertebrae number to [0,1], we have 25 vertebrae
            vertebrae_normalized = vertebrae_num / 25
            # get random image from tensor dict as we assume all images have the same shape
            random_key = list(tensors.keys())[0]
            tensors['vertebrae'] = torch.full((1, tensors[random_key].shape[1], tensors[random_key].shape[2]), vertebrae_normalized)
        
        if self.crop_mask:
            # dont input the mask thats outside the tissue
            mask = tensors['mask']
            tissue = tensors['tissue']
            new_mask = torch.where(tissue > 0, mask, torch.zeros_like(mask))
            tensors['mask'] = new_mask           

        # Combine inputs into a single tensor based on desired input
        input_tensor = torch.cat([tensors[key] for key in self.desired_input], dim=0) # Shape: [len(desired_input
        # Combine outputs into a single tensor based on desired output
        output_tensor = torch.cat([tensors[key] for key in self.desired_output], dim=0)
        #if vertebrate is the only desired output
        if self.desired_output == ['vertebrae']:
            # only return a single value
            return input_tensor, torch.tensor(vertebrae_num)
        
        return input_tensor, output_tensor
 
    def split_data(self,hydra_output_dir, train_size=0.8, val_size=0.2,augmentations = [], seed=None):
        """
        Split the data into training and validation sets
        Creates a new folder in the hydra output directory with the split data
        Returns two new instances of the BaseClass class with the split data        
        """        
        assert seed is not None, "Seed is required"
        np.random.seed(seed)
        # Shuffle the identifiers
        np.random.shuffle(self.identifers)
        # Split the identifiers
        train_size = int(len(self.identifers) * train_size)
        val_size = len(self.identifers) - train_size
        train_identifiers = self.identifers[:train_size]
        val_identifiers = self.identifers[train_size:]

        # save the validation data
        val_data_dir = os.path.join(hydra_output_dir, 'val_data')
        os.makedirs(val_data_dir, exist_ok=True)
        # make subdirectories for each type of data

        for identifier in val_identifiers:
            for dir in self.dirs:
                # make the subdirectories
                sub_dir = os.path.join(val_data_dir, dir)                
                os.makedirs(os.path.join(val_data_dir, dir), exist_ok=True)

                if dir == 'vertebrae':
                    filename = dir + '_' + identifier + '.txt'
                else:
                    filename = dir + '_' + identifier + '.png'

                src_path = os.path.join(getattr(self, dir + '_dir'), filename)
                dst_path = os.path.join(sub_dir, filename)
                shutil.copy(src_path, dst_path)

        # save the training data (potentially augmented)
        train_data_dir = os.path.join(hydra_output_dir, 'train_data')
        os.makedirs(train_data_dir, exist_ok=True)
        
        if augmentations:
            new_identifer = 0
            for identifier in train_identifiers:
                path_to_files = {f"{dir}_path": os.path.join(getattr(self, dir + '_dir'),dir + "_" + identifier + ".png") for dir in self.dirs}
                if 'vertebrae' in self.dirs:
                    path_to_files['vertebrae_path'] = path_to_files['vertebrae_path'].replace('.png', '.txt')
                           
                images = {dir.split("_")[0]: Image.open(path).convert('L') for dir, path in path_to_files.items() if dir != 'vertebrae_path'}                            
                # copy of images without the corrupted image
                images_copy = images.copy()
                images_copy.pop('corrupted')

                for augmentation_class_instance in augmentations:                    
                    augmented_images = augmentation_class_instance.augmentation(**images_copy)
                    for image_dict in augmented_images:
                        # save the augmented images in the train_data directory
                        for image_type,image in image_dict.items():
                            sub_dir = os.path.join(train_data_dir, image_type)
                            os.makedirs(sub_dir, exist_ok=True)
                            filename = image_type + '_' + str(new_identifer) + '.png'
                            dst_path = os.path.join(sub_dir, filename)
                            image.save(dst_path)
                        #also save the vertebrae file and ct file                        
                        vertebrae_src_path = path_to_files['vertebrae_path']
                        vertebrae_dst_path_dir = os.path.join(train_data_dir, 'vertebrae')
                        os.makedirs(vertebrae_dst_path_dir, exist_ok=True)
                        vertebrae_dst_path = os.path.join(train_data_dir + '/vertebrae', 'vertebrae_' + f"{new_identifer}.txt")
                        shutil.copy(vertebrae_src_path, vertebrae_dst_path) 

                        ct_src_path = path_to_files['ct_path']
                        ct_dst_path_dir = os.path.join(train_data_dir, 'ct')
                        os.makedirs(ct_dst_path_dir, exist_ok=True)
                        ct_dst_path = os.path.join(train_data_dir + '/ct', 'ct_' + f"{new_identifer}.png")
                        shutil.copy(ct_src_path, ct_dst_path)                                            

                        new_identifer += 1
        else:
            # just copy the files
            for identifier in train_identifiers:
                for dir in self.dirs:
                    # make the subdirectories
                    sub_dir = os.path.join(train_data_dir, dir)
                    os.makedirs(sub_dir, exist_ok=True)

                    if dir == 'vertebrae':
                        filename = dir + '_' + identifier + '.txt'
                    else:
                        filename = dir + '_' + identifier + '.png'

                    src_path = os.path.join(getattr(self, dir + '_dir'), filename)
                    dst_path = os.path.join(sub_dir, filename)
                    shutil.copy(src_path, dst_path)
        
        print(f"Data split and saved in {hydra_output_dir}")

        # Create new instances of the BaseClass class with the split data
        train_dataset = BaseClass(train_data_dir, transform=self.transform, dirs=self.dirs, desired_input=self.desired_input, desired_output=self.desired_output, crop_mask=self.crop_mask)
        val_dataset = BaseClass(val_data_dir, transform=self.transform, dirs=self.dirs, desired_input=self.desired_input, desired_output=self.desired_output, crop_mask=self.crop_mask)

        return train_dataset, val_dataset








  





 # Define the Dataset class
class CTInpaintingDatasetInitial(Dataset):
    def __init__(self, data_dir, transform=None):
        self.corrupted_dir = os.path.join(data_dir, 'corrupted')
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.tissue_dir = os.path.join(data_dir, 'tissue')
        self.vertebrae_dir = os.path.join(data_dir, 'vertebrae')
        self.ct_dir = os.path.join(data_dir, 'ct')

        # Get list of corrupted images
        self.filenames = sorted(os.listdir(self.corrupted_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Get the filename
        filename = self.filenames[idx]  # e.g., 'corrupted_000_1.png'

        # Extract patient ID and slice number
        # Remove the prefix 'corrupted_' and the extension '.png'
        base_filename = filename[len('corrupted_'):-len('.png')]  # '000_1'
        patient_id_str, slice_num_str = base_filename.split('_')
        patient_id = int(patient_id_str)
        slice_num = int(slice_num_str)

        # Construct file paths
        corrupted_path = os.path.join(self.corrupted_dir, filename)
        mask_filename = filename.replace('corrupted_', 'mask_')
        mask_path = os.path.join(self.mask_dir, mask_filename)
        tissue_filename = filename.replace('corrupted_', 'tissue_')
        tissue_path = os.path.join(self.tissue_dir, tissue_filename)
        ct_filename = filename.replace('corrupted_', 'ct_')
        ct_path = os.path.join(self.ct_dir, ct_filename)
        vertebrae_filename = filename.replace('corrupted_', 'vertebrae_').replace('.png', '.txt')
        vertebrae_path = os.path.join(self.vertebrae_dir, vertebrae_filename)

        # Load images
        corrupted = Image.open(corrupted_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        tissue = Image.open(tissue_path).convert('L')
        ct = Image.open(ct_path).convert('L')

        # Read the vertebrae number from the .txt file
        with open(vertebrae_path, 'r') as f:
            vertebrae_num = int(f.read().strip())
        # Normalize the vertebrae number to [0,1]
        vertebrae_normalized = (vertebrae_num - 1) / (33 - 1)  # Assuming vertebrae numbers from 1 to 33

        if self.transform:
            corrupted = self.transform(corrupted)  # Shape: [1, H, W]
            mask = self.transform(mask)
            tissue = self.transform(tissue)
            ct = self.transform(ct)
            # Create vertebrae_tensor with same H and W
            H, W = corrupted.shape[1], corrupted.shape[2]
            vertebrae_tensor = torch.full((1, H, W), vertebrae_normalized)
        else:
            assert False, "Transform is required"

        # Combine inputs into a single tensor
        input_tensor = torch.cat([corrupted, mask, tissue, vertebrae_tensor], dim=0)  # Shape: [4, H, W]

        return input_tensor, ct
    
if __name__ == "__main__":
    # Define the data directory
    data_dir = 'CT_Inpainting/data'

    # Define the transformation
    transform = transforms.ToTensor()
    # Create the dataset
    dataset = BaseClass(data_dir, transform=transform)

    """
    # Get the first sample
    input_tensor, ct = dataset[0]

    # Print the shapes of the input tensor and the CT image
    print('Input tensor shape:', input_tensor.shape)  # Shape: [4, H, W]
    print('CT image shape:', ct.shape)  # Shape: [H, W]

    # Plot the input tensor and the CT image
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i in range(4):
        axs[i].imshow(input_tensor[i].numpy(), cmap='gray')
        axs[i].axis('off')
    axs[4].imshow(ct.numpy(), cmap='gray')
    axs[4].axis('off')
    plt.tight_layout()
    plt.show()
    """