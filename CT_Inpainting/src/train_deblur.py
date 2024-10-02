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
from src.model_classes.model import VGG19Features, PerceptualLoss
from src.data.data_set_classes import BaseClass
from src.data.data_set_augmentations import flipMaskAug
import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
from torch.optim.lr_scheduler import StepLR
from src.model_classes.model import UNet


def area_to_fill_mask(mask, tissue):
    """
    This function takes in a mask and tissue image and returns the area to fill in the mask
    """
    # check that the mask only contrains 0 and 1
    assert torch.all((mask == 0) | (mask == 1)), "Mask must only contain 0 and 1"
    fill_mask = torch.where(tissue > 0, mask, torch.zeros_like(mask))
    return fill_mask


@hydra.main(version_base=None, config_path="deblur_config", config_name="base_config")
def train(cfg: DictConfig):

    # Debug mode
    debug = cfg.debug
    seed = cfg.training_params.seed
    num_epochs = cfg.training_params.num_epochs
    learning_rate = cfg.training_params.learning_rate
    batch_size = cfg.training_params.batch_size
    gen_model_dir = cfg.training_params.gen_model_dir    
    loss_functions = cfg.training_params.loss_functions
    clamp_output = cfg.training_params.clamp_output
    use_scheduler = cfg.training_params.use_scheduler
    vgg_layers = cfg.training_params.vgg_layers
    perceptual_loss_weight = cfg.training_params.perceptual_loss_weight

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    assert cfg.training_params.loss_functions[0] == "l1", "First loss function must be L1"
    base_criterion = nn.L1Loss()   
    # Create an instance of the UNet model with only 1 input channel,
    #  which will be the generated image from our generative model
    model = UNet(in_channels=2, out_channels=1)      
    model.to(device)
    
    #optimizer, we could experiment with different optimizers and add to config
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)   
    if use_scheduler:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # get where hydra is storing everything, whis is specified in the config
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['run']['dir']     

    # wandb parameters
    api_key = cfg.wandb.api_key
    wandb.login(key=api_key)
    # Initialize Weights and Biases

    if debug:
        name_for_wandb = "debug" + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        name_for_wandb = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    wandb.init(
        project=cfg.wandb.project,
        name = name_for_wandb,
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,       
            "seed": seed,
            "debug": debug,
            "model": "UNet_deblur",
            "architecture": "UNet_deblur",
            "dataset": "Deblur",
            
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
    # load the training data
    train_dir = os.path.join(os.path.join(gen_model_dir, "train_data"))
    val_dir = os.path.join(os.path.join(gen_model_dir, "val_data"))    
    # dirs is only generated data
    sub_data_dirs = ["generated","tissue","ct"]    

    train_dataset = BaseClass(data_dir=train_dir, transform=transform,dirs=sub_data_dirs,desired_input=["generated","tissue"])
    val_dataset = BaseClass(data_dir=val_dir, transform=transform,dirs=sub_data_dirs,desired_input=["generated","tissue"])

    # if true, use small dataset for testing/debug
    if debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(8))
        val_dataset = torch.utils.data.Subset(val_dataset, range(2))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
       
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Training loop with progress bars and W&B logging
    best_val_loss = float('inf')
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
                if clamp_output:
                    outputs = torch.clamp(outputs, 0, 1)
           
                # Calculate loss
                l1_loss = base_criterion(outputs, labels)
                
                total_loss = l1_loss

                total_loss.backward()
                optimizer.step()

                total_train_loss += total_loss.item() * inputs.size(0)
                MAE_train_loss += l1_loss.item() * inputs.size(0)

                # Update progress bar for each batch
                train_bar.set_postfix(loss=total_loss.item())
                train_bar.update(1)
        if use_scheduler:

            scheduler.step()

        # Calculate average training loss
        total_train_loss = total_train_loss / len(train_loader.dataset)
        MAE_train_loss = MAE_train_loss / len(train_loader.dataset)

        # Log the training loss to W&B
        wandb.log({"epoch": epoch + 1, "total_train_loss": total_train_loss})   
       
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
                    if clamp_output:
                        outputs = torch.clamp(outputs, 0, 1)

                    # Calculate loss
                    loss = base_criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

                    # Update progress bar for each batch
                    val_bar.set_postfix(loss=loss.item())
                    val_bar.update(1)

                    # Visualize the first few reconstructed images and ground truth
                    if batch_idx == 0:  # Visualize the first batch only                       
                        batch_size_plotting = inputs.size(0)  # Get the actual batch size
                        columns = 4  # Number of images to display per row (corrupted, mask, reconstructed, ground truth)
                        rows = batch_size  # One row per batch sample

                        fig, axs = plt.subplots(rows, columns, figsize=(15, 5 * batch_size_plotting))

                        for i in range(batch_size_plotting):                            
                            inputs_np = inputs[i, 0].cpu().numpy()  # Corrupted image for each sample
                            tissue = inputs[i, 1].cpu().numpy()  # Mask image for each sample
                            reconstructed_np = outputs[i, 0].cpu().numpy()  # Reconstructed image for each sample
                            ground_truth_np = labels[i, 0].cpu().numpy()  # Ground truth image for each sample

                            # Plotting for each sample
                            axs[i, 0].imshow(inputs_np, cmap='gray')
                            axs[i, 0].set_title(f'Blurred Image (Sample {i+1})')
                            axs[i, 0].axis('off')       

                            axs[i, 1].imshow(tissue, cmap='gray')
                            axs[i, 1].set_title('Tissue')
                            axs[i, 1].axis('off')

                            axs[i, 2].imshow(reconstructed_np, cmap='gray')
                            axs[i, 2].set_title(f'Reconstructed (Epoch {epoch+1})')
                            axs[i, 2].axis('off')

                            axs[i, 3].imshow(ground_truth_np, cmap='gray')
                            axs[i, 3].set_title('Ground Truth')
                            axs[i, 3].axis('off')

                        # Adjust layout
                        plt.tight_layout()

                        # Save the whole batch figure
                        plt.savefig(f'{output_dir}/epoch_{epoch+1}_batch_reconstruction.png')

                        # Convert matplotlib figure to a numpy array
                        fig.canvas.draw()
                        combined_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        combined_image = combined_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                        # Log the combined image to W&B
                        wandb.log({"epoch": epoch + 1, "batch_comparison_images": wandb.Image(combined_image, caption=f"Epoch {epoch+1} Batch Comparison")})

                        # Close the plot to free up memory
                        plt.close(fig)

        val_loss = val_loss / len(val_loader.dataset)

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #torch.save(model.state_dict(), 'models/best_ct_inpainting_unet.pth')
            torch.save(model.state_dict(), f'{output_dir}/best_model.pth')        

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