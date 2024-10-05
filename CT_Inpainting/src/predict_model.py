import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.utils import l1_score, plot_prediction, load_sample
from src.model_classes.model import UNet  # Assuming you have the UNet model in src.models.model
#from src.train_model import CTInpaintingDataset  # Import the dataset class from train.py
from torchvision import transforms

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model prediction function
def predict(corrupted_image: np.ndarray, 
            tissue_image: np.ndarray,
            mask_image: np.ndarray,
            vertebrae_number: int, 
            model: torch.nn.Module) -> np.ndarray:
    """
    Predict using the trained model.
    
    Args:
    - corrupted_image (np.ndarray): Corrupted input CT image of shape (H, W).
    - tissue_image (np.ndarray): Tissue image of shape (H, W).
    - mask_image (np.ndarray): Mask image of shape (H, W).
    - vertebrae_number (int): Integer representing the vertebrae number of the slice.
    - model (torch.nn.Module): Trained PyTorch model.
    
    Returns:
    - np.ndarray: Reconstructed image of shape (H, W).
    """

    # Define the transformations

    
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(lambda x: x / 255.0)
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # convert numpy arrays to tensors and add channel dimension
    corrupted = torch.tensor(corrupted_image, dtype=torch.float32,device=device) 
    mask = torch.tensor(mask_image, dtype=torch.float32, device=device)        
    tissue = torch.tensor(tissue_image, dtype=torch.float32, device=device)
    
    # NOTE, after competetion end we notice
    # that the vertebrae_number is normalized differently here compared to the training
    # left unchanged such that results of our submition can be recreated. 
    vertebrae_normalized = (vertebrae_number - 1) / (33 - 1)  # Assuming vertebrae numbers from 1 to 33

    if transform:
        corrupted = transform(corrupted)[None, ...]  # Shape: [1, H, W]
        mask = transform(mask)[None, ...]
        tissue = transform(tissue)[None, ...]
        # Create vertebrae_tensor with same H and W
        H, W = corrupted.shape[1], corrupted.shape[2]
        vertebrae_tensor = torch.full((1, H, W), vertebrae_normalized, device=device, dtype=torch.float32)  # Shape: [1, H, W]

    # Combine inputs into a single tensor
    input_tensor = torch.cat([corrupted, mask, tissue, vertebrae_tensor], dim=0)  # Shape: [4, H, W]

    # Make predictions using the model
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output_tensor = model(input_tensor[None, ...])  # Get the model output
    
    # lets try clamping the output tensor
    output_tensor = torch.clamp(output_tensor, 0, 1)

    reconstructed_np = output_tensor[0, 0].detach().cpu().numpy() 
    reconstructed_image = reconstructed_np * 255.0   

    return reconstructed_image


def apply_only_to_mask(corrupted_image:np.ndarray, 
                tissue_image:np.ndarray,
                mask_image:np.ndarray,
                reconstructed_image:np.ndarray,
                ) -> np.ndarray:
    
    final_image = corrupted_image.astype(float)
    fill_mask = (tissue_image>0) & (mask_image>0)
    final_image[fill_mask] = reconstructed_image[fill_mask]
    return final_image


