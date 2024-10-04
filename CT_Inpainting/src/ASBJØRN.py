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
import datetime
import torch.nn as nn
import torch

# Torch clear cache
torch.cuda.empty_cache()

########## Helper Functions ##########
def visualize_and_log(inputs, outputs, labels, epoch):
    # Extracting individual components from the inputs tensor
    corrupted_image_np = inputs[0, 0].cpu().numpy()   # Corrupted Image
    mask_np = inputs[0, 1].cpu().numpy()              # Mask
    mask_cropped_np = inputs[0, 2].cpu().numpy()      # Cropped Mask
    tissue_np = inputs[0, 3].cpu().numpy()            # Tissue

    reconstructed_np = outputs[0, 0].cpu().numpy()
    ground_truth_np = labels[0, 0].cpu().numpy()

    # Create a figure with 6 subplots
    fig, axs = plt.subplots(1, 6, figsize=(24, 5))
    
    axs[0].imshow(corrupted_image_np, cmap='gray')
    axs[0].set_title('Corrupted Image')
    axs[0].axis('off')

    axs[1].imshow(mask_np, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')

    axs[2].imshow(mask_cropped_np, cmap='gray')
    axs[2].set_title('Cropped Mask')
    axs[2].axis('off')

    axs[3].imshow(tissue_np, cmap='gray')
    axs[3].set_title('Tissue')
    axs[3].axis('off')

    axs[4].imshow(reconstructed_np, cmap='gray')
    axs[4].set_title(f'Reconstructed (Epoch {epoch + 1})')
    axs[4].axis('off')

    axs[5].imshow(ground_truth_np, cmap='gray')
    axs[5].set_title('Ground Truth')
    axs[5].axis('off')

    plt.tight_layout()
    plt.savefig(f'plots/epoch_{epoch + 1}_reconstruction_new.png')
    plt.show()

    # Convert the figure to a format that can be logged in W&B
    fig.canvas.draw()
    combined_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    combined_image = combined_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Log to W&B
    wandb.log({
        "epoch": epoch + 1,
        "comparison_images": wandb.Image(combined_image, caption=f"Epoch {epoch + 1} Comparison")
    })

    plt.close(fig)


def save_model(model):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'models/ct_inpainting_unet_{timestamp}.pth')



########## Dataset class ##########
class CTInpaintingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.corrupted_dir = os.path.join(data_dir, "corrupted")
        self.mask_dir = os.path.join(data_dir, "mask")
        self.tissue_dir = os.path.join(data_dir, "tissue")
        self.vertebrae_dir = os.path.join(data_dir, "vertebrae")
        self.ct_dir = os.path.join(data_dir, "ct")

        # Get list of corrupted images
        self.filenames = sorted(os.listdir(self.corrupted_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # Construct file paths
        mask_filename = filename.replace("corrupted_", "mask_")
        mask_path = os.path.join(self.mask_dir, mask_filename)
        tissue_filename = filename.replace("corrupted_", "tissue_")
        tissue_path = os.path.join(self.tissue_dir, tissue_filename)
        ct_filename = filename.replace("corrupted_", "ct_")
        ct_path = os.path.join(self.ct_dir, ct_filename)
        vertebrae_filename = filename.replace("corrupted_", "vertebrae_").replace(".png", ".txt")
        vertebrae_path = os.path.join(self.vertebrae_dir, vertebrae_filename)

        # Load images
        mask = Image.open(mask_path).convert("L")
        tissue = Image.open(tissue_path).convert("L")
        ct = Image.open(ct_path).convert("L")
        
        with open(vertebrae_path, "r") as f:
            vertebrae_num = int(f.read().strip())
            
        # Normalize the vertebrae number to [0,1]
        vertebrae_normalized = (vertebrae_num - 1) / (25 - 1)  # Assuming vertebrae numbers from 1 to 33

        if self.transform:
            # Apply transforms to CT and mask_cropped first
            ct = self.transform(ct)
            
            import pdb; pdb.set_trace()
            # Randomly rotate the mask by 0, 90, 180, or 270 degrees
            num_rotations = np.random.randint(0, 4)  # Randomly choose 0, 1, 2, or 3 rotations (90, 180, 270 degrees)
            tissue = self.transform(tissue)
            mask = self.transform(mask)
    
            mask_rotated = torch.rot90(mask, num_rotations, [1, 2])

            
            tissue_np = np.array(tissue)
            mask_np = np.array(mask_rotated)
            mask_np[tissue_np == 0] = 0
            mask_cropped = Image.fromarray(mask_np)
            
            mask_cropped = self.transform(mask_cropped)
            tissue = self.transform(tissue)
            
            
            # Create the new corrupted image based on the rotated mask and ground truth CT
            corrupted_np = ct.numpy().copy()  # Use ground truth to recompute corrupted
            mask_rotated_np = mask_rotated.numpy()
            corrupted_np[mask_rotated_np == 1] = 0  # Apply mask on the ground truth
            corrupted_new = torch.tensor(corrupted_np).float()  # Convert back to tensor

            # Create vertebrae_tensor with the same H and W
            H, W = corrupted_new.shape[1], corrupted_new.shape[2]
            vertebrae_tensor = torch.full((1, H, W), vertebrae_normalized)


        else:
            raise ValueError("Transform is required")

        # Combine inputs into a single tensor
        input_tensor = torch.cat([corrupted_new, mask_rotated, mask_cropped, tissue, vertebrae_tensor], dim=0)  # Shape: [5, H, W]

        return input_tensor, ct

########## Model ##########
class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super(UNet, self).__init__()
        # Encoder
        self.enc_conv0 = self.conv_block(in_channels, 64)
        self.pool0 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv1 = self.conv_block(64, 128)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv2 = self.conv_block(128, 256)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv3 = self.conv_block(256, 512)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv4 = self.conv_block(512, 1024)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Additional encoder block to make it deeper
        self.enc_conv5 = self.conv_block(1024, 2048)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec_conv4 = self.conv_block(2048, 1024)

        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv3 = self.conv_block(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(512, 256)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv1 = self.conv_block(256, 128)

        self.upconv0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv0 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_prob=0.1):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_prob),
        )
        return block

    def forward(self, x):
        # Encoder
        enc0 = self.enc_conv0(x)
        enc0_pool = self.pool0(enc0)

        enc1 = self.enc_conv1(enc0_pool)
        enc1_pool = self.pool1(enc1)

        enc2 = self.enc_conv2(enc1_pool)
        enc2_pool = self.pool2(enc2)

        enc3 = self.enc_conv3(enc2_pool)
        enc3_pool = self.pool3(enc3)

        enc4 = self.enc_conv4(enc3_pool)
        enc4_pool = self.pool4(enc4)

        bottleneck = self.enc_conv5(enc4_pool)

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec_conv4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec_conv3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec_conv2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec_conv1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, enc0), dim=1)
        dec0 = self.dec_conv0(dec0)

        out = self.final_conv(dec0)
        return out

if __name__ == "__main__":
  ########## Instantiate Data, Model and Hyperparameters ##########
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  transform = transforms.Compose([transforms.Resize((256, 256)),
                                  transforms.ToTensor()])

  data_dir = 'data'  
  dataset = CTInpaintingDataset(data_dir=data_dir, transform=transform)

  train_size = int(0.8 * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


  #train_dataset = torch.utils.data.Subset(train_dataset, range(20))
  #val_dataset = torch.utils.data.Subset(val_dataset, range(4))

  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


  model = UNet().to(device)
  criterion = nn.L1Loss()  # MAE # Maybe try Huber loss
  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  api_key = "c187178e0437c71d461606e312d20dc9f1c6794f"

  wandb.login(key=api_key)
  wandb.init(
      project="CT_Inpainting",  # Replace with your project name
      config={
          "learning_rate": 2e-4,
          "epochs": 1000,
          "batch_size": 16,
          "architecture": "UNet",
          "dataset": "CT Inpainting",
      }
  )


  ########## Training ##########
  best_val_loss = float('inf')  # Initialize the best validation loss to infinity
  best_model_path = 'models/ASBJØRN_LAST_ATTEMPT_BEST_NEW.pth'  # Path to save the best model
  for epoch in range(1000):
      print(f'Epoch {epoch + 1}/{1000}')

      # Training phase
      model.train()
      train_loss = 0.0

      with tqdm(total=len(train_loader), desc="Training", unit="batch") as train_bar:
          for inputs, labels in train_loader:
              inputs, labels = inputs.to(device), labels.to(device)

              optimizer.zero_grad()
              outputs = model(inputs)
              loss = criterion(outputs, labels)

              loss.backward()
              optimizer.step()

              train_loss += loss.item() * inputs.size(0)
              train_bar.set_postfix(loss=loss.item())
              train_bar.update(1)

      train_loss /= train_size
      wandb.log({"epoch": epoch + 1, "train_loss": train_loss})

      
      # Validation phase
      model.eval()
      val_loss = 0.0

      with tqdm(total=len(val_loader), desc="Validation", unit="batch") as val_bar:
          with torch.no_grad():
              for batch_idx, (inputs, labels) in enumerate(val_loader):
                  inputs, labels = inputs.to(device), labels.to(device)

                  outputs = model(inputs)
                  loss = criterion(outputs, labels)
                  val_loss += loss.item() * inputs.size(0)

                  val_bar.set_postfix(loss=loss.item())
                  val_bar.update(1)

                  if epoch % 2 == 0:
                    # Visualize the first batch
                    if batch_idx == 0:
                        visualize_and_log(inputs, outputs, labels, epoch)

      val_loss /= val_size
      wandb.log({"epoch": epoch + 1, "val_loss": val_loss})

      # Check if this is the best model so far
      if val_loss < best_val_loss:
          best_val_loss = val_loss
          print(f"New best model found at epoch {epoch + 1} with validation loss {val_loss:.4f}. Saving model.")
          torch.save(model.state_dict(), best_model_path)  # Save the model

      
      print(f'Epoch {epoch + 1} - Training Loss: {train_loss:.4f}') # , Validation Loss: {val_loss:.4f}')

  # Finish W&B run
  wandb.finish()

  # Save the final model with timestamp (optional)
  save_model(model)