import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from collections import Counter
import numpy as np
from models.model import SimpleClassifier, Autoencoder, ClassifierOnAE
from predict_model import predict_local
from data.make_dataset import LoadTifDataset, PadToSize

# Function to train the model
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

# Define the padding size
target_width = 1500
target_height = 1470

train_transform = transforms.Compose([
    PadToSize(target_width=target_width, target_height=target_height),
    # transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet mean and std

])

# Define transformations for validation data (no augmentation)
val_transform = transforms.Compose([
    PadToSize(target_width=target_width, target_height=target_height),
    # transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet mean and std

])

image_dir = "data/training"
csv_file_path = "data/training.csv"
train_dataset = LoadTifDataset(image_dir=image_dir, csv_file_path=csv_file_path, transform=train_transform)
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
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.model.fc.parameters(), lr=0.0001)  # Reduced learning rate for fine-tuning

# Train the model
num_epochs = 50
model = train_model(model, train_dataloader, criterion, optimizer, device, num_epochs=num_epochs)

score = predict_local(model, train_dataloader, calculate_custom_score, device)
print(score)



################## AE
# Loss function and optimizer
autoencoder = Autoencoder(latent_dim=256).to(device)
criterion = nn.MSELoss()  # Use MSE loss for image reconstruction
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    autoencoder.train()
    running_loss = 0.0
    for inputs, _ in train_dataloader:  # No need for labels when training the autoencoder
        inputs = inputs.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs, latent_space = autoencoder(inputs)
        loss = criterion(outputs, inputs)  # Reconstruction loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)}")

# Training the classifier using latent space representations
classifier = ClassifierOnAE(latent_dim=256, num_classes=2).to(device)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
classifier_criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    classifier.train()
    running_loss = 0.0
    for inputs, labels in train_dataloader:  # Use only labeled data here
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Get the latent space representation from the autoencoder
        _, latent_space = autoencoder(inputs)
        
        # Train the classifier on latent space
        outputs = classifier(latent_space)
        loss = classifier_criterion(outputs, labels)
        classifier_optimizer.zero_grad()
        loss.backward()
        classifier_optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Classifier Loss: {running_loss / len(train_dataloader)}")

