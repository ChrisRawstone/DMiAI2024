import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import wandb  # Import wandb for Weights and Biases integration
from tqdm import tqdm  # Import tqdm for progress bars
import matplotlib.pyplot as plt  # Import matplotlib for visualization
import numpy as np
from src.model_classes.model import VGG19Features, PerceptualLoss
from src.data.data_set_classes import BaseClass
from src.data.data_set_augmentations import flipMaskAug, randomMaskAug
import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
from torch.optim.lr_scheduler import StepLR


def area_to_fill_mask(mask, tissue):
    """
    This function takes in a mask and tissue image and returns the area to fill in the mask
    """
    # check that the mask only contrains 0 and 1
    assert torch.all((mask == 0) | (mask == 1)), "Mask must only contain 0 and 1"
    fill_mask = torch.where(tissue > 0, mask, torch.zeros_like(mask))
    return fill_mask


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
    train_size_proportion = cfg.training_params.train_size
    augmentations_list = cfg.training_params.augmentations
    crop_mask = cfg.training_params.crop_mask
    only_score_within_mask = cfg.training_params.only_score_within_mask
    clamp_output = cfg.training_params.clamp_output
    use_scheduler = cfg.training_params.use_scheduler
    if "proportion_to_leave_unchanged" in cfg.training_params:
        proportion_to_leave_unchanged = cfg.training_params.proportion_to_leave_unchanged
    else:
        proportion_to_leave_unchanged = 0.0
    # check if training_params.use_attention is in the config
    if "use_attention" in cfg.training_params:
        use_attention = cfg.training_params.use_attention
    else:
        use_attention = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if "unet_2" in cfg.training_params:
        if cfg.training_params.unet_2:
            # overwrite import of model
            print("Using UNet version 2")
            from src.model_classes.model_2 import UNet
            print("Using UNet version 2")
            model = UNet().to(device)
        else:
            from src.model_classes.model import UNet
            model = UNet(use_attention=use_attention).to(device)
    else:
        from src.model_classes.model import UNet
        model = UNet(use_attention=use_attention).to(device)

    augmentations = []
    if augmentations_list:        
        for aug in augmentations_list:
            if aug == "flipMaskAug":
                augmentations.append(flipMaskAug())
            elif aug == "randomMaskAug":
                augmentations.append(randomMaskAug())
            else:
                raise ValueError(f"Unknown augmentation {aug}")
    
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert cfg.training_params.loss_functions[0] == "l1", "First loss function must be L1"
    base_criterion = nn.L1Loss()
    if "perceptual" in cfg.training_params.loss_functions:
        perceptual_loss_fn = PerceptualLoss(layers=vgg_layers).to(device)
    # add more loss functions here if needed    
    
    #optimizer, we could experiment with different optimizers and add to config
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)   
    if use_scheduler:
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

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
            "vgg_layers": vgg_layers,
            "perceptual_loss_weight": perceptual_loss_weight,
            "seed": seed,
            "debug": debug,
            "model": "UNet",
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
    # dont pass portion to leave unchanged here, since we want to use the same dataset for training and validation
    dataset = BaseClass(data_dir=data_dir, transform=transform, crop_mask=crop_mask)


    # Split dataset into training and validation sets
    train_size = int(train_size_proportion * len(dataset))
    val_size = len(dataset) - train_size

    # if true, use small dataset for testing/debug
    if debug:
        train_size_proportion = 0.01

    # split the dataset into training and validation sets usng our custom split_data method if augmentations
    
    train_dataset,val_dataset =dataset.split_data(output_dir, train_size=train_size_proportion, val_size=1-train_size_proportion, seed=seed, augmentations=augmentations, proportion_to_leave_unchanged=proportion_to_leave_unchanged)
    
    
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
                if "perceptual" in cfg.training_params.loss_functions:   
                    if only_score_within_mask:                        
                        # only score within the mask
                        mask = inputs[:, 1, :, :].unsqueeze(1) # important to unsqueeze to get the right shape 
                        # now only score within the mask
                        area_to_fill =area_to_fill_mask(mask, labels)
                        outputs = torch.where(area_to_fill > 0, outputs, torch.zeros_like(outputs))

                        labels = torch.where(area_to_fill > 0, labels, torch.zeros_like(labels))
                        # plot and save to make sure it is working
                        out_for_viz = outputs[0, 0].cpu().detach().numpy()
                        lab_for_viz = labels[0, 0].cpu().detach().numpy()
                        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
                        axs[0].imshow(out_for_viz, cmap='gray')
                        axs[0].set_title('predicted filling')
                        axs[0].axis('off')
                        axs[1].imshow(lab_for_viz, cmap='gray')
                        axs[1].set_title('ground truth of area to fill')
                        axs[1].axis('off')
                        plt.tight_layout()
                        plt.savefig(f'{output_dir}/epoch_{epoch+1}_only_score_within_mask.png')

                    # turn into 3 channels for perceptual loss since it expects 3 channels RGB             

                    outputs_for_perceptual_loss = torch.cat([outputs, outputs, outputs], dim=1)                    
                    labels_for_perceptual_loss = torch.cat([labels, labels, labels], dim=1)                      

                    # calculate perceptual loss                       
                    perceptual_loss = perceptual_loss_fn(outputs_for_perceptual_loss, labels_for_perceptual_loss)
                    # combine the two losses
                    total_loss = l1_loss + perceptual_loss_weight * perceptual_loss
                    print(f"l1_loss: {l1_loss.item()}, perceptual_loss: {perceptual_loss.item()}")
                else:
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
                            mask_np = inputs[i, 1].cpu().numpy()  # Mask image for each sample
                            reconstructed_np = outputs[i, 0].cpu().numpy()  # Reconstructed image for each sample
                            ground_truth_np = labels[i, 0].cpu().numpy()  # Ground truth image for each sample

                            # Plotting for each sample
                            axs[i, 0].imshow(inputs_np, cmap='gray')
                            axs[i, 0].set_title(f'Corrupted Image (Sample {i+1})')
                            axs[i, 0].axis('off')

                            axs[i, 1].imshow(mask_np, cmap='gray')
                            axs[i, 1].set_title('Mask')
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

        val_loss = val_loss / val_size

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
        torch.save(model.state_dict(), f'{output_dir}/debug.pth')
    else:
        # get time stamp
        
        # Save the trained model
        #torch.save(model.state_dict(), f'models/ct_inpainting_unet_{timestamp}.pth')
        torch.save(model.state_dict(), f'{output_dir}/epoch_{epoch+1}.pth')


if __name__ == "__main__":
    train()