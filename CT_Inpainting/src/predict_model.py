import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.utils import l1_score, plot_prediction, load_sample
from src.models.model import UNet  # Assuming you have the UNet model in src.models.model
from src.train_model import CTInpaintingDataset  # Import the dataset class from train.py
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

    reconstructed_np = output_tensor[0, 0].detach().cpu().numpy() 
    reconstructed_image = (reconstructed_np - np.min(reconstructed_np)) / (np.max(reconstructed_np) - np.min(reconstructed_np)) * 255.0

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




# Main function to run the prediction and visualization
def main():
    # Set the path to the data directory
    data_dir = 'data'  # Adjust this to your dataset path

    # Load the dataset using the CTInpaintingDataset class
    # Define the transformations

    PATIENT_IX = "000_0"

    sample = load_sample(PATIENT_IX)
    tissue_image = sample["tissue_image"]
    corrupted_image = sample["corrupted_image"]
    mask_image = sample["mask_image"]
    ct_image = sample["ct_image"]
    vertebrae = sample["vertebrae"]




    model = UNet().to(device)
    model.load_state_dict(torch.load('models/ct_inpainting_unet_20240928_162225.pth', map_location=device))  # Replace with your model path
    model.to(device)




    reconstructed_image = predict(corrupted_image, tissue_image, mask_image, vertebrae, model)
    
    reconstructed_image = apply_only_to_mask(corrupted_image, tissue_image, mask_image, reconstructed_image)
    # Load the trained model (Ensure you have the model weights file in the correct path)


    # scale back to 0-255 using min max scaling

    # reconstructed_image = reconstructed_image.astype(np.uint8)



    # give me max value of array
    # print(np.max(reconstructed_image))

    # Predict reconstruction using the loaded model
    # output = predict(corrupted_image, tissue_image, mask_image, vertebrae_number, model)
    # reconstructed_image = output[0,0]


    plt.imshow(reconstructed_image , cmap='gray')
    # plt.imshow(ct_image[0].cpu().numpy() , cmap='gray')
    # plt.set_title(f'Reconstructed)')
    plt.axis('off')
    plt.savefig('plots/reconstructed_image.png')

    # Plot and score the prediction
    plot_prediction(corrupted_image, tissue_image, mask_image, reconstructed_image, vertebrae, ct_image, name="model_prediction.jpg")
    
    # Calculate L1 score
    l1 = l1_score(ct_image, reconstructed_image)
    print(f"L1 score: {l1:.03f}") 

# Run the main function if this script is executed  
if __name__ == "__main__":
    main()
