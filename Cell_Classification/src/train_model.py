import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from collections import Counter
import numpy as np
from models.model import SimpleClassifier
from predict_model import predict_local
from data.make_dataset import LoadTifDataset, PadToSize
import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'I am on the device: {device}')

def calculate_custom_score(a_0, a_1, n_0, n_1):
    # Ensure no division by zero
    if n_0 == 0 or n_1 == 0:
        return 0
    return (a_0 * a_1) / (n_0 * n_1)

def train_model(model, train_dataloader, criterion, optimizer, device, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training phase
        for inputs, labels in train_dataloader:
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

            running_loss += loss.item() * inputs.size(0)  # Multiply by batch size

        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # # Evaluation phase on training set
        # model.eval()
        # total_0 = 0
        # total_1 = 0
        # correct_0 = 0
        # correct_1 = 0
        
        # with torch.no_grad():
        #     for inputs, labels in train_dataloader:
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)

        #         outputs = model(inputs)
        #         _, predicted = torch.max(outputs, 1)
        #         # Calculate correct predictions and totals for each class
        #         total_0 += (labels == 0).sum().item()
        #         total_1 += (labels == 1).sum().item()

        #         correct_0 += ((predicted == 0) & (labels == 0)).sum().item()
        #         correct_1 += ((predicted == 1) & (labels == 1)).sum().item()
                
        #         print(f'labels:{labels}')
        #         print(f'predicted:{predicted}')

        # # Calculate the custom score
        # custom_score = calculate_custom_score(correct_0, correct_1, total_0, total_1)
        # print(f"Custom Score after Epoch {epoch + 1}: {custom_score:.4f}")
        
    print("Training complete!")
    
    return model

final_resize_size = (224, 224)
# train_transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=3), 
#     transforms.Resize(final_resize_size),  # Resize the shorter side first
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
# ])

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.Resize(final_resize_size),  # Resize the shorter side first

    # Data Augmentation
    transforms.RandomHorizontalFlip(p=0.5),  # Flip image horizontally with a 50% probability
    transforms.RandomVerticalFlip(p=0.5),    # Flip image vertically with a 50% probability
    transforms.RandomRotation(degrees=15),   # Rotate image by a random angle up to 15 degrees
    transforms.RandomResizedCrop(final_resize_size, scale=(0.8, 1.0)),  # Randomly crop and resize

    # Color jitter for brightness/contrast adjustments (useful for microscopy images)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    
    # Convert to tensor and normalize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])


val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.Resize(final_resize_size),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize as per ImageNet
])

image_dir = "data/training/"
csv_file = "data/training.csv"

# Create datasets
train_dataset = LoadTifDataset(csv_file=csv_file, image_dir=image_dir, transform=val_transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Get the labels from the dataset
labels = train_dataset.labels_df.iloc[:, 1].values  # Assuming labels are in the second column

# Compute class counts
class_sample_counts = np.array([np.sum(labels == i) for i in range(2)])
n_samples = sum(class_sample_counts)

# Compute class weights
class_weights = n_samples / (2 * class_sample_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

model = SimpleClassifier().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights) # weight=class_weights
optimizer = optim.Adam(model.model.fc.parameters(), lr=0.0001)  # Reduced learning rate for fine-tuning
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Train the model
num_epochs = 100
model = train_model(model, train_dataloader, criterion, optimizer, device, num_epochs=num_epochs)
torch.save(model.state_dict(), 'trained_model_cell.pth')
score_train = predict_local(model, train_dataloader, calculate_custom_score, device)
print(score_train)


image_dir = "data/validation"
csv_file_path = "data/validation.csv"
val_dataset = LoadTifDataset(csv_file=csv_file_path, image_dir=image_dir,transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)
score_val = predict_local(model, val_dataloader, calculate_custom_score, device)
print(score_val)

