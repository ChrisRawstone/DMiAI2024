import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

        # Open the image as greyscale and convert to RGB
        image = Image.open(img_path).convert("L")  # Open as greyscale (L mode)
        image = image.convert("RGB")  # Convert to 3-channel RGB for ResNet compatibility

        label = int(self.annotations.iloc[index, 1])  # Convert label to int (0 or 1)

        if self.transform:
            image = self.transform(image)
        return image, label, img_path  # Return the image path as well

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        probs = torch.sigmoid(inputs)
        BCE_loss = torch.where(targets == 1, BCE_loss * self.alpha, BCE_loss)
        focal_loss = (1 - probs) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Define the main function
def main():
    # Hyperparameters
    num_epochs = 20
    batch_size = 4
    learning_rate = 1e-5  # Decreased learning rate for more stable training
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    train_dir = "data/training"
    train_csv_path = "data/training.csv"
    validation_dir = "data/validation"  # Same as in predict.py
    val_csv_path = "data/validation.csv"  # Same as in predict.py

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load annotations
    train_annotations = pd.read_csv(train_csv_path)
    val_annotations = pd.read_csv(val_csv_path)

    # Verify labels
    print("Unique labels in training set:", train_annotations.iloc[:, 1].unique())
    print("Unique labels in validation set:", val_annotations.iloc[:, 1].unique())

    # Print number of labels in each class for training and validation sets
    train_class_counts = train_annotations.iloc[:, 1].value_counts()
    val_class_counts = val_annotations.iloc[:, 1].value_counts()
    print(f"\nTraining set class distribution:\n{train_class_counts}")
    print(f"\nValidation set class distribution:\n{val_class_counts}")

    # Create datasets
    train_dataset = CustomImageDataset(
        annotations_df=train_annotations, root_dir=train_dir, transform=transform
    )
    val_dataset = CustomImageDataset(
        annotations_df=val_annotations, root_dir=validation_dir, transform=transform
    )

    # Create dataloaders without WeightedRandomSampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load pre-trained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Updated for torchvision >=0.13

    # Modify the first convolutional layer to accept single-channel (greyscale) images
    model.conv1 = nn.Conv2d(3, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify the final fully connected layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Binary classification

    # Initialize weights and biases of the final layer
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)

    # Calculate pos_weight
    num_neg = train_class_counts[0]
    num_pos = train_class_counts[1]
    pos_weight_value = num_neg / num_pos
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    print(f"Using pos_weight: {pos_weight.item():.4f}")

    # Define weighted loss
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Alternatively, use Focal Loss
    criterion = FocalLoss(alpha=pos_weight_value, gamma=2, reduction='mean')

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.4)

    # Optionally, implement a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    model = model.to(device)
    best_combined_correct = 0
    best_epoch = -1
    patience = 5
    counter = 0
    best_val_loss = float('inf')

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

            # Backward pass and optimization with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels, img_paths in val_loader:
                images, labels = images.to(device), labels.float().to(device)

                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_running_loss += loss.item()

                # Apply sigmoid to outputs and get predictions
                probs = torch.sigmoid(outputs.squeeze())
                preds = (probs >= 0.5).long()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_running_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        roc_auc = roc_auc_score(all_labels, all_probs)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), "best_resnet50_homogeneity_detection.pth")
            print(f"Saved best model at epoch {epoch+1} with validation loss {avg_val_loss:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training completed. Best model saved with validation loss {best_val_loss:.4f}.")

    # Save the trained model
    torch.save(model.state_dict(), "resnet50_homogeneity_detection.pth")

if __name__ == "__main__":
    main()
