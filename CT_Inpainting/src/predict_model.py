import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import l1_score, plot_prediction, load_sample
from src.models.model import UNet  # Assuming you have the UNet model in src.models.model
from src.train_model import CTInpaintingDataset  # Import the dataset class from train.py

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
    # Normalize vertebrae number to [0, 1]
    vertebrae_normalized = (vertebrae_number - 1) / (33 - 1)  # Assuming vertebrae numbers from 1 to 33

    

    # Convert numpy arrays to tensors and add channel dimension
    corrupted_tensor = torch.tensor(corrupted_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    mask_tensor = torch.tensor(mask_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)            # Shape: [1, 1, H, W]
    tissue_tensor = torch.tensor(tissue_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)        # Shape: [1, 1, H, W]
    
    # Create a tensor for vertebrae image with the same H and W
    H, W = corrupted_image.shape
    vertebrae_tensor = torch.full((1, 1, H, W), vertebrae_normalized, dtype=torch.float32)           # Shape: [1, 1, H, W]

    # Concatenate tensors to create a single input tensor
    input_tensor = torch.cat([corrupted_tensor, mask_tensor, tissue_tensor, vertebrae_tensor], dim=1)  # Shape: [1, 4, H, W]

    # Make predictions using the model
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output_tensor = model(input_tensor.to(device))  # Get the model output

    # Convert output tensor back to numpy array
    reconstructed_image = output_tensor.squeeze().cpu().numpy()  # Remove batch and channel dimensions

    return reconstructed_image

# Main function to run the prediction and visualization
def main():
    # Set the path to the data directory
    data_dir = 'data'  # Adjust this to your dataset path

    # Load the dataset using the CTInpaintingDataset class
    dataset = CTInpaintingDataset(data_dir=data_dir)

    # Select a sample from the dataset (use the first sample for demonstration)
    sample = dataset[0]
    
    # Extract individual components from the sample
    input_tensor, ct_image = sample
    corrupted_image = input_tensor[0].numpy()  # Corrupted image
    mask_image = input_tensor[1].numpy()       # Mask image
    tissue_image = input_tensor[2].numpy()     # Tissue image
    vertebrae_tensor = input_tensor[3].numpy()  # Vertebrae image
    vertebrae_number = int(vertebrae_tensor.mean() * (33 - 1) + 1)  # Reverse normalization to get vertebrae number

    # Load the trained model (Ensure you have the model weights file in the correct path)
    model = UNet(in_channels=4, out_channels=1)
    model.load_state_dict(torch.load('models/ct_inpainting_unet_10.pth', map_location=device))  # Replace with your model path
    model.to(device)

    # Predict reconstruction using the loaded model
    reconstructed_image = predict(corrupted_image, tissue_image, mask_image, vertebrae_number, model)

    plt.imshow(reconstructed_image, cmap='gray')
    # plt.set_title(f'Reconstructed)')
    plt.axis('off')
    plt.savefig('plots/reconstructed_image.png')

    # Plot and score the prediction
    # plot_prediction(corrupted_image, tissue_image, mask_image, reconstructed_image, vertebrae_number, ct_image.numpy())
    
    # Calculate L1 score
    l1 = l1_score(ct_image.numpy(), reconstructed_image)
    print(f"L1 score: {l1:.03f}") 

# Run the main function if this script is executed  
if __name__ == "__main__":
    main()
