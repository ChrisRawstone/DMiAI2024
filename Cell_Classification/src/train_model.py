import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

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
        return image, label, img_path  # Return the image path as well

# Define the main function
def main():
    # Hyperparameters
    num_epochs = 20
    batch_size = 4
    learning_rate = 0.000001  # Increased learning rate
    pos_weight = 1.0  # Increased pos_weight to give more weight to minority class

    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Print number of labels in each class for training and validation sets
    train_class_counts = train_df.iloc[:, 1].value_counts()
    val_class_counts = val_df.iloc[:, 1].value_counts()
    print(f"\nTraining set class distribution:\n{train_class_counts}")
    print(f"\nValidation set class distribution:\n{val_class_counts}")

    # Create datasets
    train_dataset = CustomImageDataset(
        annotations_df=train_df, root_dir=data_dir, transform=transform
    )
    val_dataset = CustomImageDataset(
        annotations_df=val_df, root_dir=data_dir, transform=transform
    )

    # Create weighted sampler for training to handle class imbalance
    class_sample_counts = train_df.iloc[:, 1].value_counts().tolist()
    weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float)
    sample_weights = weights[train_df.iloc[:, 1].values]
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Modify the final fully connected layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Binary classification

    # Define weighted loss
    # Set a higher weight for minority class
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    model = model.to(device)
    # Add this before the training loop to track best performance
    best_combined_correct = 0
    best_epoch = -1

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels, _ in train_loader:  # No need for image paths in training
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
        class_correct = {0: 0, 1: 0}  # To track number of correct predictions for each class
        max_samples_per_class = 5  # Plot 5 images per class
        images_to_plot = {0: [], 1: []}  # To store image paths per class

        with torch.no_grad():
            for images, labels, img_paths in val_loader:
                images, labels = images.to(device), labels.float().to(device)

                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_running_loss += loss.item()

                # Apply sigmoid to outputs and get predictions
                probs = torch.sigmoid(outputs.squeeze())
                preds = (probs >= 0.5).long()

                # Print out probabilities for debugging
                print(f"Predicted probabilities: {probs.tolist()}")

                # Count the number of correct predictions for each class
                for i in range(images.size(0)):
                    label = int(labels[i].item())
                    pred = int(preds[i].item())

                    # Check if the prediction is correct
                    if pred == label:
                        class_correct[label] += 1  # Increment correct counter for the class

                    img_path = img_paths[i]  # Directly get image path from batch

                    if class_samples[label] < max_samples_per_class:
                        images_to_plot[label].append((img_path, label, pred))
                        class_samples[label] += 1
                    if class_samples[0] >= max_samples_per_class and class_samples[1] >= max_samples_per_class:
                        break
                if class_samples[0] >= max_samples_per_class and class_samples[1] >= max_samples_per_class:
                    break

        val_loss = val_running_loss / len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Print number of correct predictions for each class
        print(f"Correct Predictions for Epoch {epoch+1}: Class 0: {class_correct[0]} / {val_class_counts[0]}, Class 1: {class_correct[1]} / {val_class_counts[1]}")

        # Save the model if it has the best combined correct predictions and at least 1 correct from each class
        combined_correct = class_correct[0] + class_correct[1]
        if combined_correct > best_combined_correct and class_correct[0] > 0 and class_correct[1] > 0:
            best_combined_correct = combined_correct
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "best_resnet50_homogeneity_detection.pth")
            print(f"Saved best model at epoch {epoch+1} with {combined_correct} correct predictions")

        # Consolidate all images into one plot per epoch
        fig, axes = plt.subplots(2, max_samples_per_class, figsize=(15, 6))
        for idx, label in enumerate([0, 1]):
            for jdx in range(max_samples_per_class):
                if jdx < len(images_to_plot[label]):  # Check if we have enough samples
                    img_path, true_label, pred_label = images_to_plot[label][jdx]

                    # Use matplotlib.image to read and plot .tif images
                    img = mpimg.imread(img_path)
                    axes[idx, jdx].imshow(img)
                    axes[idx, jdx].axis('off')
                    axes[idx, jdx].set_title(f'True: {true_label}, Pred: {pred_label}')

        # Adjust spacing to minimize white space
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, hspace=0.1, wspace=0.1)

        plt.suptitle(f'Epoch {epoch+1} Validation Results')
        plt.tight_layout()

        # Save the entire figure once per epoch with minimal white space
        plt.savefig(os.path.join(plot_dir, f'epoch_{epoch+1}_validation_results.png'), bbox_inches='tight')
        # plt.show()

    print(f"Training completed. Best model saved at epoch {best_epoch} with {best_combined_correct} correct predictions.")

    # Save the trained model
    torch.save(model.state_dict(), "resnet50_homogeneity_detection.pth")

if __name__ == "__main__":
    main()
