import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.diffusionModel import CTInpaintingModel
from src.data.diffusion_dataset import CTInpaintingDiffusionDataset  
from src.data.diffusion_dataset import CTInpaintingDiffusionDataset2



# Hyperparameters
batch_size = 4
num_epochs = 10
learning_rate = 1e-4

# Data preparation
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
train_dataset = CTInpaintingDiffusionDataset(data_dir='data', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = CTInpaintingModel().to("cuda")
criterion = nn.MSELoss()  
optimizer = torch.optim.AdamW(
    list(model.unet.parameters()), 
    lr=learning_rate
)
# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  # Set the model to training mode
        
        for batch in dataloader:
            # Get input batch from the dataset (input_tensor and ground truth CT)
            original_image, mask, ground_truth_ct = batch
            
            
            # Move to GPU
            original_image = original_image.to("cuda")
            mask_image = mask.to("cuda")
            ground_truth_ct = ground_truth_ct.to("cuda")
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            generator = torch.Generator("cuda").manual_seed(92)
            prompt = "restore the corrupted areas of the CT slice to get a complete image"
            output_image = model(prompt=prompt, image=original_image, mask_image=mask_image, generator=generator)
                       
                       
            # Convert outputs (generated image) back to tensor if needed
            outputs_tensor = transforms.ToTensor()(output_image).unsqueeze(0).to("cuda")  # [batch, 1, 256, 256]
            
            # Compute loss
            loss = criterion(outputs_tensor, ground_truth_ct)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track running loss
            running_loss += loss.item()
        
        # Log the loss for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")
    
    print("Training completed")

# Start training
train_model(model, train_dataloader, criterion, optimizer, num_epochs)
