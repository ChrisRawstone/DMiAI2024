import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import wandb  # Import wandb for Weights and Biases integration
from tqdm import tqdm  # Import tqdm for progress bars
import matplotlib.pyplot as plt  # Import matplotlib for visualization
import numpy as np
from src.models.model import UNet, Discriminator, UNetWithAttention, PatchDiscriminator
import datetime


# Define the Dataset class
class CTInpaintingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.corrupted_dir = os.path.join(data_dir, 'corrupted')
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.tissue_dir = os.path.join(data_dir, 'tissue')
        self.vertebrae_dir = os.path.join(data_dir, 'vertebrae')
        self.ct_dir = os.path.join(data_dir, 'ct')
        self.filenames = sorted(os.listdir(self.corrupted_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        base_filename = filename[len('corrupted_'):-len('.png')]
        patient_id_str, slice_num_str = base_filename.split('_')
        patient_id, slice_num = int(patient_id_str), int(slice_num_str)

        corrupted = Image.open(os.path.join(self.corrupted_dir, filename)).convert('L')
        mask = Image.open(os.path.join(self.mask_dir, filename.replace('corrupted_', 'mask_'))).convert('L')
        tissue = Image.open(os.path.join(self.tissue_dir, filename.replace('corrupted_', 'tissue_'))).convert('L')
        ct = Image.open(os.path.join(self.ct_dir, filename.replace('corrupted_', 'ct_'))).convert('L')

        with open(os.path.join(self.vertebrae_dir, filename.replace('corrupted_', 'vertebrae_').replace('.png', '.txt')), 'r') as f:
            vertebrae_num = int(f.read().strip())
        vertebrae_normalized = (vertebrae_num - 1) / (33 - 1)

        if self.transform:
            corrupted, mask, tissue, ct = self.transform(corrupted), self.transform(mask), self.transform(tissue), self.transform(ct)
            vertebrae_tensor = torch.full((1, corrupted.shape[1], corrupted.shape[2]), vertebrae_normalized)
        else:
            raise ValueError("Transform is required")

        input_tensor = torch.cat([corrupted, mask, tissue, vertebrae_tensor], dim=0)
        return input_tensor, ct

# Define the Perceptual Loss using VGG16
class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15, 22]):  # Use relu2_2 by default
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:23].eval()  # Only use first 23 layers (up to relu3_3)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.layers = layers
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_features = self.get_features(x)
        y_features = self.get_features(y)
        loss = 0.0
        for xf, yf in zip(x_features, y_features):
            loss += self.criterion(xf, yf)
        return loss

    def get_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features


def train():
  
    # Initialize the W&B run
    wandb.init()

    # Fetch the sweep config from wandb
    config = wandb.config

    num_epochs = 40
    base_lr = config.base_lr
    max_lr = config.max_lr
    batch_size = config.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations and prepare dataset
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = CTInpaintingDataset(data_dir='data', transform=transform)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize models, loss functions, and optimizers
    generator = UNetWithAttention().to(device)
    discriminator = PatchDiscriminator().to(device)
    
    # Initialize Perceptual Loss
    perceptual_loss_fn = PerceptualLoss().to(device)
    
    criterion_L1 = nn.L1Loss()  # Reconstruction loss
    criterion_BCE = nn.BCELoss()  # Adversarial loss

    # Optimizers for generator and discriminator
    optimizer_G = optim.Adam(generator.parameters(), lr=base_lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=base_lr)

    # Cyclic learning rate scheduler
    scheduler_G = optim.lr_scheduler.CyclicLR(optimizer_G, base_lr=base_lr, max_lr=max_lr, step_size_up=2000, mode='triangular')
    scheduler_D = optim.lr_scheduler.CyclicLR(optimizer_D, base_lr=base_lr, max_lr=max_lr, step_size_up=2000, mode='triangular')

    real_label, fake_label = 1.0, 0.0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        generator.train()
        discriminator.train()

        train_loss_G, train_loss_D = 0.0, 0.0

        with tqdm(total=len(train_loader), desc="Training", unit="batch") as train_bar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # -----------------------------------
                # 1. Train the Discriminator (PatchGAN)
                # -----------------------------------
                optimizer_D.zero_grad()

                # Real images
                real_output = discriminator(labels)
                real_loss = criterion_BCE(real_output, torch.ones_like(real_output) * real_label)

                # Fake images generated by U-Net
                fake_images = generator(inputs)
                fake_output = discriminator(fake_images.detach())  # Detach to avoid training generator
                fake_loss = criterion_BCE(fake_output, torch.zeros_like(fake_output) * fake_label)

                # Total discriminator loss
                loss_D = (real_loss + fake_loss) / 2
                loss_D.backward()
                optimizer_D.step()

                # -----------------------------------
                # 2. Train the Generator (U-Net with Attention and Perceptual Loss)
                # -----------------------------------
                optimizer_G.zero_grad()

                # Adversarial loss (from PatchGAN)
                adversarial_loss = criterion_BCE(discriminator(fake_images), torch.ones_like(fake_output) * real_label)

                # L1 loss
                l1_loss = criterion_L1(fake_images, labels)

                # Perceptual loss (requires 3-channel input, so we expand to 3 channels)
                fake_images_expanded = fake_images.expand(-1, 3, -1, -1)
                labels_expanded = labels.expand(-1, 3, -1, -1)
                perceptual_loss = perceptual_loss_fn(fake_images_expanded, labels_expanded)

                # Total generator loss: weighted sum of L1 loss, perceptual loss, and adversarial loss
                loss_G = l1_loss + config.adversarial_loss_weight * adversarial_loss + config.perceptual_loss_weight * perceptual_loss
                loss_G.backward()
                optimizer_G.step()

                # Step the learning rate scheduler
                scheduler_G.step()
                scheduler_D.step()

                # Accumulate losses
                train_loss_G += loss_G.item() * inputs.size(0)
                train_loss_D += loss_D.item() * inputs.size(0)
                train_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
                train_bar.update(1)

        train_loss_G /= len(train_loader.dataset)
        train_loss_D /= len(train_loader.dataset)

        # -----------------------------------
        # Validation phase
        # -----------------------------------
        generator.eval()
        val_loss = 0.0

        with tqdm(total=len(val_loader), desc="Validation", unit="batch") as val_bar:
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = generator(inputs)
                    loss = criterion_L1(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_bar.set_postfix(loss=loss.item())
                    val_bar.update(1)

        val_loss /= len(val_loader.dataset)

        # Logging to Weights and Biases
        wandb.log({
            "epoch": epoch + 1,
            "train_loss_G": train_loss_G,
            "train_loss_D": train_loss_D,
            "val_loss": val_loss
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(), 'models/best_ct_inpainting_unet_with_attention_sweep.pth')

        print(f'Epoch {epoch+1}/{num_epochs} - Training Loss G: {train_loss_G:.4f}, Training Loss D: {train_loss_D:.4f}, Validation Loss: {val_loss:.4f}')


# Example Sweep Configuration
sweep_config = {
    'method': 'bayes',  # You can also choose 'grid' or 'random'
    'metric': {
        'name': 'val_loss',  # We will optimize for validation loss
        'goal': 'minimize'
    },
    'parameters': {
        'base_lr': {
            'distribution': 'log_uniform',
            'min': 1e-6,
            'max': 1e-4
        },
        'max_lr': {
            'distribution': 'log_uniform',
            'min': 1e-3,
            'max': 1e-1
        },
        'batch_size': {
            'values': [4, 8, 16]
        },
        'adversarial_loss_weight': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.2
        },
        'perceptual_loss_weight': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.2
        }
    }
}

if __name__ == "__main__":
    # Initialize sweep and start it
    sweep_id = wandb.sweep(sweep_config, project="CT_Inpainting")
    wandb.agent(sweep_id, function=train)
