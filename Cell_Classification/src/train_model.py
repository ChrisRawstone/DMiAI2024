import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from models.model import SimpleClassifier
from data.make_dataset import LoadTifDataset, PadToSize

# Function to train the model
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=5):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")
        
    print("Training complete!")

# Load data 

# Define the padding size
target_width = 1500
target_height = 1470

# Define transformations
transform = transforms.Compose([
    PadToSize(target_width=target_width, target_height=target_height),  # Custom padding transform
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()])

image_dir = "data/training"
csv_file_path = "data/training.csv"
dataset = LoadTifDataset(image_dir=image_dir, csv_file_path=csv_file_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleClassifier(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, dataloader, criterion, optimizer, device, num_epochs=5)