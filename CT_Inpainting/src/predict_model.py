# predict_model.py

import torch
import numpy as np
from src.utils import l1_score, plot_prediction, load_sample
from src.train_model import UNet  # Import the UNet model


# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_predict(corrupted_image: np.ndarray, 
                  tissue_image: np.ndarray,
                  mask_image: np.ndarray,
                  vertebrae: int,
                  model: torch.nn.Module) -> np.ndarray:
    """
    Predict using the trained model.
    
    Args:
    - corrupted_image (np.ndarray): Corrupted input CT image of shape (H, W).
    - tissue_image (np.ndarray): Tissue image of shape (H, W).
    - mask_image (np.ndarray): Mask image of shape (H, W).
    - vertebrae (int): Integer representing the vertebrae number of the slice.
    - model (torch.nn.Module): Trained PyTorch model.
    
    Returns:
    - np.ndarray: Reconstructed image of shape (H, W).
    """
    # Normalize vertebrae number to [0, 1]
    vertebrae_normalized = (vertebrae - 1) / (33 - 1)  # Assuming vertebrae numbers from 1 to 33

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

    # reconstructed_image = np.clip(reconstructed_image, 0, 255)

    # reconstructed_image = reconstructed_image.astype(np.uint8)


    return reconstructed_image

def main():
    # Define the patient index to load
    PATIENT_IX = "000_0"

    # Load the sample data for the specified patient
    sample = load_sample(PATIENT_IX)

    # Extract the required images and vertebrae number from the sample
    tissue_image = sample["tissue_image"]
    corrupted_image = sample["corrupted_image"]
    mask_image = sample["mask_image"]
    ct_image = sample["ct_image"]  # Ground truth image
    vertebrae = sample["vertebrae"]




    # Initialize or load the trained model
    model = UNet(in_channels=4, out_channels=1)  # Create an instance of the UNet model
    model.load_state_dict(torch.load("models/ct_inpainting_unet.pth", map_location=device))  # Load trained weights
    model.to(device)

    # Predict reconstruction using the model
    reconstructed_image = model_predict(corrupted_image, tissue_image, mask_image, vertebrae, model)

    # Plot and score the prediction
    plot_prediction(corrupted_image, tissue_image, mask_image, reconstructed_image, vertebrae, ct_image)
    l1 = l1_score(ct_image, reconstructed_image)  # Calculate L1 score between ground truth and reconstructed image
    print(f"L1 score: {l1:.03f}")

if __name__ == "__main__":
    main()
