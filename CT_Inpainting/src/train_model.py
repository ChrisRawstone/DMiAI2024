import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import wandb
from tqdm import tqdm
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
        
        # Load images
        corrupted = Image.open(os.path.join(self.corrupted_dir, filename)).convert('L')
        mask = Image.open(os.path.join(self.mask_dir, filename.replace('corrupted_', 'mask_'))).convert('L')
        tissue = Image.open(os.path.join(self.tissue_dir, filename.replace('corrupted_', 'tissue_'))).convert('L')
        ct = Image.open(os.path.join(self.ct_dir, filename.replace('corrupted_', 'ct_'))).convert('L')

        # Load vertebrae number and normalize
        with open(os.path.join(self.vertebrae_dir, filename.replace('corrupted_', 'vertebrae_').replace('.png', '.txt')), 'r') as f:
            vertebrae_num = int(f.read().strip())
        vertebrae_normalized = (vertebrae_num - 1) / 32  # Normalized assuming vertebrae 1-33
        
        if self.transform:
            corrupted = self.transform(corrupted)
            mask = self.transform(mask)
            tissue = self.transform(tissue)
            ct = self.transform(ct)
            vertebrae_tensor = torch.full((1, corrupted.shape[1], corrupted.shape[2]), vertebrae_normalized)
        else:
            raise ValueError("Transform is required")

        # Combine inputs
        input_tensor = torch.cat([corrupted, mask, tissue, vertebrae_tensor], dim=0)
        return input_tensor, ct

# Define the Perceptual Loss using VGG16
class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15, 22]):  # Default layers
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:23].eval()  # Up to relu4_3
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.layers = layers
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_features = self.get_features(x)
        y_features = self.get_features(y)
        loss = sum(self.criterion(xf, yf) for xf, yf in zip(x_features, y_features))
        return loss

    def get_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

def main():
    # Config
    num_epochs = 40
    base_lr = 1e-5
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize W&B
    wandb.login(key="c187178e0437c71d461606e312d20dc9f1c6794f")
    wandb.init(
        project="CT_Inpainting",
        config={"learning_rate": base_lr, "epochs": num_epochs, "batch_size": batch_size, "architecture": "UNetWithAttention"}
    )

    # Data preparation
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = CTInpaintingDataset(data_dir='data', transform=transform)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, optimizer
    generator = UNetWithAttention().to(device)
    perceptual_loss_fn = PerceptualLoss().to(device)
    criterion_L1 = nn.L1Loss()
    optimizer_G = optim.Adam(generator.parameters(), lr=base_lr)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        generator.train()

        total_train_loss = 0.0

        # Training phase
        with tqdm(total=len(train_loader), desc="Training", unit="batch") as train_bar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer_G.zero_grad()

                # Forward pass
                fake_images = generator(inputs)

                # L1 loss
                l1_loss = criterion_L1(fake_images, labels)

                # Perceptual loss
                fake_images_expanded = fake_images.expand(-1, 3, -1, -1)
                labels_expanded = labels.expand(-1, 3, -1, -1)
                perceptual_loss = perceptual_loss_fn(fake_images_expanded, labels_expanded)

                # Total loss
                loss_G = l1_loss + 0.01 * perceptual_loss
                loss_G.backward()
                optimizer_G.step()

                # Accumulate loss
                total_train_loss += loss_G.item() * inputs.size(0)
                train_bar.set_postfix(loss_G=loss_G.item())
                train_bar.update(1)

        total_train_loss /= len(train_loader.dataset)

        # Validation phase
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

        # Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "Total_train_loss": total_train_loss,
            "val_loss": val_loss
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            torch.save(generator.state_dict(), f'models/best_ct_inpainting_unet_with_attention_{timestamp}.pth')

        print(f'Epoch {epoch+1}/{num_epochs} - Training Loss: {total_train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    main()
