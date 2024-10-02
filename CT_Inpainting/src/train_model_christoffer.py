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
from src.model_classes.model import UNet
import datetime


def main():
    num_epochs = 100  # Adjust the number of epochs as needed
    learning_rate=1e-3
    batch_size = 8
    api_key = "c187178e0437c71d461606e312d20dc9f1c6794f"
    data_dir = 'CT_Inpainting/data_sorted_by_vertebrae/0'  # Adjust this path to your data directory

    # use small dataset for testing
    test_small_dataset = False

    wandb.login(key=api_key)

    # Initialize Weights and Biases
    wandb.init(
        project="CT_Inpainting",  # Replace with your project name
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "architecture": "UNet",
            "dataset": "CT Inpainting",
        }
    )

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    data_inputs = torch.load('CT_Inpainting/data_sorted_by_vertebrae/0/tensor_data_set.pt')
    data_labels = torch.load('CT_Inpainting/data_sorted_by_vertebrae/0/tensor_label_set.pt')

    # Create a custom dataset
    class CTDataset(Dataset):
        def __init__(self, inputs, labels, transform=None):
            self.inputs = inputs
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            input = self.inputs[idx]
            ct = self.labels[idx]

            if self.transform:
                input = self.transform(input)
                ct = self.transform(ct)               
            return input, ct
    
    dataset = CTDataset(data_inputs, data_labels, transform=None)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # if true, use small dataset for testing
    if test_small_dataset:
        train_dataset = torch.utils.data.Subset(train_dataset, range(4))
        val_dataset = torch.utils.data.Subset(val_dataset, range(2))


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    #criterion = nn.L1Loss()  # MAE
    criterion = nn.L1Loss(reduction='none')  # MAE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with progress bars and W&B logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_loss = 0.0

        # Use tqdm to add a progress bar to the training loop
        with tqdm(total=len(train_loader), desc="Training", unit="batch") as train_bar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device)  # inputs shape: [batch_size, 4, 256, 256]
                # visualize the input
                plt.imshow(inputs[0,0,:,:].cpu().numpy())
                # create directory if it does not exist
                os.makedirs('CT_Inpainting/plots', exist_ok=True)
                plt.savefig('CT_Inpainting/plots/input.png')
                labels = labels.to(device)  # labels shape: [batch_size, 1, 256, 256]
                # they do not have that dim, so add the channel dim
                labels = labels.unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(inputs)

                #loss = criterion(outputs, labels)

                # find the overlap between the mask and tissue

                mask = inputs[:, 1]  # Mask image
                tissue = inputs[:, 2]
                area_to_fill = (tissue>0) & (mask>0) # get the indices where the mask and tissue overlap
                area_to_fill = area_to_fill.unsqueeze(1)
                
                # adjust the loss to only consider the area that needs to be filled

                loss = criterion(outputs[area_to_fill], labels[area_to_fill])
                # normalize the loss according to the number of pixels that needed to be filled
                # Adding a small epsilon to avoid division by zero
                epsilon = 1e-8
                loss = torch.sum(loss) / area_to_fill.sum() + epsilon
                      
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
                    # they miss the channel dim, so add it
                    labels = labels.unsqueeze(1)

                    # Perform reconstruction
                    outputs = model(inputs)

                    mask = inputs[:, 1]  # Mask image
                    tissue = inputs[:, 2]
                    area_to_fill = (tissue>0) & (mask>0) # get the indices where the mask and tissue overlap
                    area_to_fill = area_to_fill.unsqueeze(1)
                    
                    # adjust the loss to only consider the area that needs to be filled

                    loss = criterion(outputs[area_to_fill], labels[area_to_fill])
                    # normalize the loss according to the number of pixels that needed to be filled
                    # Adding a small epsilon to avoid division by zero
                    epsilon = 1e-8
                    loss = torch.sum(loss) / area_to_fill.sum() + epsilon

                    # Calculate loss
                    #loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

                    # Update progress bar for each batch
                    val_bar.set_postfix(loss=loss.item())
                    val_bar.update(1)

                    # Visualize the first few reconstructed images and ground truth
                    if batch_idx == 1:  # Visualize the first batch only
                        inputs_np = inputs[0, 0].cpu().numpy()  # Original corrupted image
                        mask_np = inputs[0, 1].cpu().numpy()  # Mask image
                        reconstructed_np = outputs[0, 0].cpu().numpy()  # Reconstructed image
                        #ground_truth_np = labels[0, 0].cpu().numpy()  # Ground truth image
                        ground_truth_np = labels[0,:,:].cpu().numpy()  # Ground truth image

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

                        # if 
                        #axs[3].imshow(ground_truth_np, cmap='gray')
                        axs[3].imshow(ground_truth_np[0], cmap='gray') # quick fix for dim mismatch
                        axs[3].set_title('Ground Truth')
                        axs[3].axis('off')

                        # Save the figure
                        plt.tight_layout()
                        plt.savefig(f'CT_Inpainting/plots/{timestamp}_epoch_{epoch+1}_reconstruction.png')
                        #plt.savefig(f'plots/epoch_{epoch+1}_reconstruction.png')
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
        # save the model every 5th epoch
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f'CT_Inpainting/models/ct_inpainting_unet_{timestamp}_epoch_{epoch+1}.pth')

    # Finish the W&B run
    wandb.finish()

    if test_small_dataset:
        torch.save(model.state_dict(), 'models/shit_ct_inpainting_unet.pth')
    else:
        # get time stamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the trained model
        torch.save(model.state_dict(), f'CT_Inpainting/models/ct_inpainting_unet_{timestamp}.pth')




if __name__ == "__main__":
    main()