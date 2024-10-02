# ct_inpainting.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import wandb  # Import wandb for Weights and Biases integration
from tqdm import tqdm  # Import tqdm for progress bars
import matplotlib.pyplot as plt  # Import matplotlib for visualization
import numpy as np
from src.model_classes.model import UNet, VGG19Features, PerceptualLoss
from data.data_set_classes import BaseClass
from src.data.data_set_augmentations import flipMaskAug
import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
from torch.optim.lr_scheduler import StepLR

@hydra.main(version_base=None, config_path="classifier_config", config_name="base_config_classifier")
def train(cfg: DictConfig):

    # Debug mode
    debug = cfg.debug

    # traing parameters
    data_dir = cfg.training_params.data_dir
    seed = cfg.training_params.seed
    num_epochs = cfg.training_params.num_epochs
    learning_rate = cfg.training_params.learning_rate
    batch_size = cfg.training_params.batch_size
    train_size_proportion = cfg.training_params.train_size
    use_pretrained = cfg.training_params.use_pretrained
    use_scheduler = cfg.training_params.use_scheduler
            
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Initialize the model, loss function, and optimizer
        # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load VGG19 model
    model = torchvision.models.vgg19(pretrained=use_pretrained).to(device)  
    # change the last layer to have 25 classes
    # 25 classes for the 25 vertebrae
    model.classifier[6] = nn.Linear(4096, 25)  
    base_criterion = nn.CrossEntropyLoss()   
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
        name_for_wandb = "debug_classifier" + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        name_for_wandb = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_classifier"

    wandb.init(
        project=cfg.wandb.project,
        name = name_for_wandb,
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "seed": seed,
            "debug": debug,
            "model": "VGG19",
            "architecture": "VGG19",
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
    dataset = BaseClass(data_dir=data_dir, transform=transform,desired_input=["ct"],desired_output=["vertebrae"])
    
    # Split dataset into training and validation sets
    train_size = int(train_size_proportion * len(dataset))
    val_size = len(dataset) - train_size

    # just split using the default split method in pytorch
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))       
    
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
       
        # Use tqdm to add a progress bar to the training loop
        with tqdm(total=len(train_loader), desc="Training", unit="batch") as train_bar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device)  # inputs shape: [batch_size, 4, 256, 256]
                labels = labels.to(device)  # labels shape: [batch_size, 1, 256, 256]

                optimizer.zero_grad()
                # VGG expects 3 channel images
                inputs = torch.cat((inputs, inputs, inputs), 1)
                outputs_logits = model(inputs)
                #softmax
                #output_probs = torch.nn.functional.softmax(outputs, dim=1)
                # Calculate loss
                #class_predictions = output_probs.argmax(dim=1)
                total_loss = base_criterion(outputs_logits, labels)

                total_loss.backward()
                optimizer.step()

                total_train_loss += total_loss.item() * inputs.size(0)       

                # Update progress bar for each batch
                train_bar.set_postfix(loss=total_loss.item())
                train_bar.update(1)
                
        if use_scheduler:
            scheduler.step()

        # Calculate average training loss
        total_train_loss = total_train_loss / train_size

        # Log the training loss to W&B
        wandb.log({"epoch": epoch + 1, "total_train_loss": total_train_loss})
        # also log accuracy
 
           
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        

        # Use tqdm to add a progress bar to the validation loop
        with tqdm(total=len(val_loader), desc="Validation", unit="batch") as val_bar:
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # VGG expects 3 channel images
                    inputs = torch.cat((inputs, inputs, inputs), 1)
                    outputs_logits = model(inputs)                    
                    # Calculate loss
                    loss = base_criterion(outputs_logits, labels)
                    val_loss += loss.item() * inputs.size(0)

                    # Get the class predictions
                    output_probs = torch.nn.functional.softmax(outputs_logits, dim=1)
                    outputs = output_probs.argmax(dim=1)
                    val_predictions.extend(outputs.cpu().detach().numpy())
                    val_labels.extend(labels.cpu().detach().numpy())

                    # Update progress bar for each batch
                    val_bar.set_postfix(loss=loss.item())
                    val_bar.update(1)

                    
        val_loss = val_loss / val_size

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #torch.save(model.state_dict(), 'models/best_ct_inpainting_unet.pth')
            torch.save(model.state_dict(), f'{output_dir}/best_model.pth')        

        correct_predictions = sum([1 for i in range(len(val_labels)) if val_labels[i] == val_predictions[i]])
        correct_predictions = correct_predictions / len(val_labels)
        # Log the validation loss to W&B
        wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "correct_predictions": correct_predictions})
        # log the labels and predictions
        
        print(f'Epoch {epoch+1}/{num_epochs} - Training Loss Total: {total_train_loss:.4f}, Correct Predictions: {correct_predictions:.4f}, Validation Loss: {val_loss:.4f}')

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