import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Custom Dataset class for loading images
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

# Define the evaluation function
def evaluate_model():
    # Paths
    validation_dir = "data/validation"  # Folder with validation images
    validation_csv = "data/validation.csv"  # CSV with validation annotations
    model_path = "resnet50_homogeneity_detection.pth"  # Path to the saved model
    plot_dir = "validation_plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image transformations (Ensure consistency with train.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load validation annotations
    annotations = pd.read_csv(validation_csv)

    # Create validation dataset
    val_dataset = CustomImageDataset(
        annotations_df=annotations, root_dir=validation_dir, transform=transform
    )

    # Create dataloader for validation set
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Load the pre-trained ResNet50 model (Consistent Initialization)
    model = models.resnet50(pretrained=True)  # Changed from pretrained=False to pretrained=True
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Binary classification

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))  # Ensure loading on the correct device
    model = model.to(device)

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Evaluation
    model.eval()
    val_running_loss = 0.0
    class_samples = {0: 0, 1: 0}  # To keep track of how many images we have per class
    class_correct = {0: 0, 1: 0}  # To track number of correct predictions for each class
    max_samples_per_class = 5  # Plot 5 images per class
    images_to_plot = {0: [], 1: []}  # To store image paths per class

    total_samples = 0  # Track the total number of samples
    total_correct = 0  # Track the total number of correct predictions

    # Custom score counters
    n_0 = 0
    n_1 = 0
    a_0 = 0
    a_1 = 0

    with torch.no_grad():
        for images, labels, img_paths in val_loader:
            images, labels = images.to(device), labels.float().to(device)

            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_running_loss += loss.item()

            # Apply sigmoid to outputs and get predictions
            probs = torch.sigmoid(outputs.squeeze())

            # Use consistent thresholding
            threshold = 0.5  # Changed from 0.3 to 0.5
            preds = (probs >= threshold).long()

            # Count the number of correct predictions for each class
            for i in range(images.size(0)):
                label = int(labels[i].item())
                pred = int(preds[i].item())

                # Increment counters
                if label == 0:
                    n_0 += 1
                else:
                    n_1 += 1

                if pred == label:
                    if label == 0:
                        a_0 += 1
                    else:
                        a_1 += 1

                # Check if the prediction is correct
                if pred == label:
                    class_correct[label] += 1  # Increment correct counter for the class
                    total_correct += 1  # Increment total correct counter

                total_samples += 1  # Increment total sample counter

                img_path = img_paths[i]  # Directly get image path from batch

                if len(images_to_plot[label]) < max_samples_per_class:
                    images_to_plot[label].append((img_path, label, pred))

    # Calculate accuracy
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Calculate the custom score
    custom_score = (a_0 * a_1) / (n_0 * n_1) if n_0 > 0 and n_1 > 0 else 0

    print(f"\nValidation Results")
    print(f"Validation Loss: {val_running_loss / len(val_loader):.4f}")
    print(f"Total Correct Predictions: {total_correct} / {total_samples}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct Predictions for Class 0: {class_correct[0]}")
    print(f"Correct Predictions for Class 1: {class_correct[1]}")
    print(f"Custom Score: {custom_score:.4f}")

    # Consolidate all images into one plot for visualization
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
    plt.suptitle(f'Validation Results - Accuracy: {accuracy:.4f} - Custom Score: {custom_score:.4f}')
    plt.tight_layout()

    # Save the entire figure once with minimal white space
    plt.savefig(os.path.join(plot_dir, f'validation_results.png'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    evaluate_model()
