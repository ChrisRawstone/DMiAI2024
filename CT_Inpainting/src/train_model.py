# ct_inpainting.py

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





# Define the Dataset class
class CTInpaintingDataset(Dataset):
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
            # Handle case without transforms (not used here)
            # If no transform is provided, just convert to tensor manually
            corrupted = torch.tensor(np.array(corrupted), dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)
            tissue = torch.tensor(np.array(tissue), dtype=torch.float32).unsqueeze(0)
            ct = torch.tensor(np.array(ct), dtype=torch.float32).unsqueeze(0)
            H, W = corrupted.shape[1], corrupted.shape[2]
            vertebrae_tensor = torch.full((1, H, W), vertebrae_normalized, dtype=torch.float32)

        # Combine inputs into a single tensor
        input_tensor = torch.cat([corrupted, mask, tissue, vertebrae_tensor], dim=0)  # Shape: [4, H, W]

        return input_tensor, ct















# Load the model for inference (if needed)
# model.load_state_dict(torch.load('ct_inpainting_unet.pth'))
# model.eval()

# Example of making predictions on new data
# def predict(model, input_tensor):
#     model.eval()
#     with torch.no_grad():
#         input_tensor = input_tensor.to(device)
#         output = model(input_tensor.unsqueeze(0))  # Add batch dimension
#         output = output.squeeze(0)  # Remove batch dimension
#         output = output.cpu()
#     return output

# To test on a single sample (replace 'sample_idx' with an index)
# sample_idx = 0
# input_tensor, _ = dataset[sample_idx]
# predicted_output = predict(model, input_tensor)
# Convert the predicted output to PIL Image and save or display
# predicted_image = transforms.ToPILImage()(predicted_output)
# predicted_image.save('predicted_ct.png')

def main():
    num_epochs = 20  # Adjust the number of epochs as needed
    learning_rate=1e-4
    batch_size = 4
    api_key = "c187178e0437c71d461606e312d20dc9f1c6794f"

    # use small dataset for testing
    test_small_dataset = False

    wandb.login(key=api_key)

    # Initialize Weights and Biases
    wandb.init(
        project="CT_Inpainting",  # Replace with your project name
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": 8,
            "architecture": "UNet",
            "dataset": "CT Inpainting",
        }
    )




    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # Converts images to tensors with values in [0, 1]
    ])

    # Prepare the dataset and dataloaders
    data_dir = 'data'  # Adjust this path to your data directory

    dataset = CTInpaintingDataset(data_dir=data_dir, transform=transform)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # if true, use small dataset for testing
    if test_small_dataset:
        train_dataset = torch.utils.data.Subset(train_dataset, range(8))
        val_dataset = torch.utils.data.Subset(val_dataset, range(2))


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = UNet().to(device)
    criterion = nn.L1Loss()  # MAE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with progress bars and W&B logging
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_loss = 0.0

        # Use tqdm to add a progress bar to the training loop
        with tqdm(total=len(train_loader), desc="Training", unit="batch") as train_bar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device)  # inputs shape: [batch_size, 4, 256, 256]
                labels = labels.to(device)  # labels shape: [batch_size, 1, 256, 256]

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

                # Update progress bar for each batch
                train_bar.set_postfix(loss=loss.item())
                train_bar.update(1)

        # Calculate average training loss
        train_loss = train_loss / train_size

        # Log the training loss to W&B
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss})

        # Validation phase
        model.eval()
        val_loss = 0.0

        # Use tqdm to add a progress bar to the validation loop
        with tqdm(total=len(val_loader), desc="Validation", unit="batch") as val_bar:
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Perform reconstruction
                    outputs = model(inputs)

                    # Calculate loss
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

                    # Update progress bar for each batch
                    val_bar.set_postfix(loss=loss.item())
                    val_bar.update(1)

                    # Visualize the first few reconstructed images and ground truth
                    if batch_idx == 0:  # Visualize the first batch only
                        inputs_np = inputs[0, 0].cpu().numpy()  # Original corrupted image
                        mask_np = inputs[0, 1].cpu().numpy()  # Mask image
                        reconstructed_np = outputs[0, 0].cpu().numpy()  # Reconstructed image
                        ground_truth_np = labels[0, 0].cpu().numpy()  # Ground truth image

                        # Plotting
                        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
                        axs[0].imshow(inputs_np, cmap='gray')
                        axs[0].set_title('Corrupted Image')
                        axs[0].axis('off')

                        axs[1].imshow(mask_np, cmap='gray')
                        axs[1].set_title('Mask')
                        axs[1].axis('off')

                        axs[2].imshow(reconstructed_np, cmap='gray')
                        axs[2].set_title(f'Reconstructed (Epoch {epoch+1})')
                        axs[2].axis('off')

                        axs[3].imshow(ground_truth_np, cmap='gray')
                        axs[3].set_title('Ground Truth')
                        axs[3].axis('off')

                        # Save the figure
                        plt.tight_layout()
                        plt.savefig(f'plots/epoch_{epoch+1}_reconstruction.png')
                        plt.show()

                        # Convert matplotlib figure to a numpy array
                        fig.canvas.draw()
                        combined_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        combined_image = combined_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                        # Log the combined image to W&B
                        wandb.log({"epoch": epoch + 1, "comparison_images": wandb.Image(combined_image, caption=f"Epoch {epoch+1} Comparison")})

                        # Close the plot to free up memory
                        plt.close(fig)

        val_loss = val_loss / val_size

        # Log the validation loss to W&B
        wandb.log({"epoch": epoch + 1, "val_loss": val_loss})

        print(f'Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Finish the W&B run
    wandb.finish()

    if test_small_dataset:
        torch.save(model.state_dict(), 'models/shit_ct_inpainting_unet.pth')
    else:   
        # Save the trained model
        torch.save(model.state_dict(), 'models/ct_inpainting_unet_10.pth')



if __name__ == "__main__":
    main()