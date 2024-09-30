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
from src.models.model import UNet, VGG19Features, PerceptualLoss
from src.data.data_set_classes import BaseClass
import datetime
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="model_config", config_name="base_config")
def train(cfg: DictConfig):

    # Debug mode
    debug = cfg.debug

    # traing parameters
    data_dir = cfg.training_params.data_dir
    seed = cfg.training_params.seed
    num_epochs = cfg.training_params.num_epochs
    learning_rate = cfg.training_params.learning_rate
    batch_size = cfg.training_params.batch_size
    vgg_layers = cfg.training_params.vgg_layers
    perceptual_loss_weight = cfg.training_params.perceptual_loss_weight

    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Initialize the model, loss function, and optimizer
        # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)

    assert cfg.training_params.loss_functions[0] == "l1", "First loss function must be L1"
    base_criterion = nn.L1Loss()
    if "perceptual" in cfg.training_params.loss_functions:
        perceptual_loss_fn = PerceptualLoss(layers=vgg_layers).to(device)
    # add more loss functions here if needed    
    
    #optimizer, we could experiment with different optimizers and add to config
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)   

    # get where hydra is storing everything, whis is specified in the config
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['run']['dir']     

    # wandb parameters
    api_key = cfg.wandb.api_key
    wandb.login(key=api_key)
    # Initialize Weights and Biases
    wandb.init(
        project=cfg.wandb.project,
        name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "architecture": "UNet",
            "dataset": "CT Inpainting",
            
        }
    )
    # get where hydra is storing everything, whis is specified in the config
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['run']['dir']
    
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # Converts images to tensors with values in [0, 1]
    ])

    # Prepare the dataset and dataloaders
    dataset = BaseClass(data_dir=data_dir, transform=transform)

    # Split dataset into training and validation sets
    train_size = int(cfg.training_params.train_size * len(dataset))
    val_size = len(dataset) - train_size

    # set seed for reproducibility when splitting the dataset
    torch.manual_seed(seed)    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # if true, use small dataset for testing/debug
    if debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(8))
        val_dataset = torch.utils.data.Subset(val_dataset, range(2))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
       
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Training loop with progress bars and W&B logging
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        total_train_loss = 0.0
        MAE_train_loss = 0.0

        # Use tqdm to add a progress bar to the training loop
        with tqdm(total=len(train_loader), desc="Training", unit="batch") as train_bar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device)  # inputs shape: [batch_size, 4, 256, 256]
                labels = labels.to(device)  # labels shape: [batch_size, 1, 256, 256]

                optimizer.zero_grad()
                outputs = model(inputs)

                #calculate perceptual loss

                l1_loss = base_criterion(outputs, labels)
                if "perceptual" in cfg.training_params.loss_functions:
                    # fix outputs such that it has 3 channels and can be used in perceptual loss
                    outputs_for_perceptual_loss = torch.cat([outputs, outputs, outputs], dim=1)
                    # same for labels
                    labels_for_perceptual_loss = torch.cat([labels, labels, labels], dim=1)                    
                    perceptual_loss = perceptual_loss_fn(outputs_for_perceptual_loss, labels_for_perceptual_loss)
                    # combine the two losses
                    total_loss = l1_loss + perceptual_loss_weight * perceptual_loss
                else:
                    total_loss = l1_loss

                total_loss.backward()
                optimizer.step()

                total_train_loss += total_loss.item() * inputs.size(0)
                MAE_train_loss += l1_loss.item() * inputs.size(0)

                # Update progress bar for each batch
                train_bar.set_postfix(loss=total_loss.item())
                train_bar.update(1)

        # Calculate average training loss
        total_train_loss = total_train_loss / train_size
        MAE_train_loss = MAE_train_loss / train_size

        # Log the training loss to W&B
        wandb.log({"epoch": epoch + 1, "total_train_loss": total_train_loss})
        if "perceptual" in cfg.training_params.loss_functions:
            wandb.log({"epoch": epoch + 1, "perceptual_loss": perceptual_loss})
       
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
                    loss = base_criterion(outputs, labels)
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
                        #plt.savefig(f'plots/epoch_{epoch+1}_reconstruction.png')
                        plt.savefig(f'{output_dir}/epoch_{epoch+1}_reconstruction.png')
                        #plt.show()

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

        print(f'Epoch {epoch+1}/{num_epochs} - Training Loss Total: {total_train_loss:.4f}, Validation Loss MAE: {val_loss:.4f}')
        # save the model every 5th epoch
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f'{output_dir}/epoch_{epoch+1}.pth')

    # Finish the W&B run
    wandb.finish()

    if debug:
        #torch.save(model.state_dict(), 'models/shit_ct_inpainting_unet.pth')
        torch.save(model.state_dict(), f'{output_dir}/debug.pth')
    else:
        # get time stamp
        #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the trained model
        #torch.save(model.state_dict(), f'models/ct_inpainting_unet_{timestamp}.pth')
        torch.save(model.state_dict(), f'{output_dir}/epoch_{epoch+1}.pth')


if __name__ == "__main__":
    train()