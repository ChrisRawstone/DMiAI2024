import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Custom Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, annotations_df, root_dir, transform=None, id_length=3):
        self.annotations = annotations_df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.id_length = id_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Format the image_id with leading zeros based on `id_length`
        img_id = str(self.annotations.iloc[index, 0]).zfill(self.id_length)
        img_path = os.path.join(self.root_dir, f"{img_id}.tif")

        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[index, 1])  # Convert label to int (0 or 1)

        if self.transform:
            image = self.transform(image)
        return image, label

# Define the main function
def main():
    # Hyperparameters
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001

    # Paths
    data_dir = "data/training"
    csv_path = "data/training.csv"
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load annotations
    annotations = pd.read_csv(csv_path)

    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(
        annotations,
        test_size=0.2,
        stratify=annotations.iloc[:, 1],  # Stratify on the labels
        random_state=42
    )

    # Create datasets
    train_dataset = CustomImageDataset(
        annotations_df=train_df, root_dir=data_dir, transform=transform
    )
    val_dataset = CustomImageDataset(
        annotations_df=val_df, root_dir=data_dir, transform=transform
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Modify the final fully connected layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Binary classification

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)  # Move to GPU and change to float

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)  # Squeeze for correct loss shape

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_running_loss = 0.0
        class_samples = {0: 0, 1: 0}  # To keep track of how many images we have per class
        max_samples_per_class = 2
        images_to_plot = {0: [], 1: []}  # To store image paths per class

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)

                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_running_loss += loss.item()

                # Apply sigmoid to outputs and get predictions
                probs = torch.sigmoid(outputs.squeeze())
                preds = (probs >= 0.5).long()

                # For plotting, collect image paths where needed
                for i in range(images.size(0)):
                    label = int(labels[i].item())
                    pred = int(preds[i].item())
                    img_id = str(val_df.iloc[i, 0]).zfill(3)
                    img_path = os.path.join(data_dir, f"{img_id}.tif")

                    if class_samples[label] < max_samples_per_class:
                        images_to_plot[label].append((img_path, label, pred))
                        class_samples[label] += 1
                    if class_samples[0] >= max_samples_per_class and class_samples[1] >= max_samples_per_class:
                        break
                if class_samples[0] >= max_samples_per_class and class_samples[1] >= max_samples_per_class:
                    break

        val_loss = val_running_loss / len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Plotting
        fig, axes = plt.subplots(2, max_samples_per_class, figsize=(10, 5))
        for idx, label in enumerate([0, 1]):
            for jdx in range(max_samples_per_class):
                img_path, true_label, pred_label = images_to_plot[label][jdx]

                # Use matplotlib.image to read and plot .tif images
                img = mpimg.imread(img_path)
                axes[idx, jdx].imshow(img)
                axes[idx, jdx].axis('off')
                axes[idx, jdx].set_title(f'True: {true_label}, Pred: {pred_label}')
                # Save individual plot images
        
        plt.suptitle(f'Epoch {epoch+1} Validation Results')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'epoch_{epoch+1}_class_{label}_img.png'))
                

        plt.show()

    print("Training completed.")

    # Save the trained model
    torch.save(model.state_dict(), "resnet50_homogeneity_detection.pth")

if __name__ == "__main__":
    main()
