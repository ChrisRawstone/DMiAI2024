# training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data.make_dataset import LoadTifDataset
from models.model import Autoencoder, Classifier
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_custom_score(a_0, a_1, n_0, n_1):
    # Ensure no division by zero
    if n_0 == 0 or n_1 == 0:
        return 0
    return (a_0 * a_1) / (n_0 * n_1)

def visualize_reconstructions(model, dataloader, device, num_images=5):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.to(device)
    with torch.no_grad():
        reconstructed = model(images)
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    for i in range(num_images):
        fig, axes = plt.subplots(1, 2)
        # Original Image
        axes[0].imshow(np.transpose(images[i], (1, 2, 0)))
        axes[0].set_title("Original")
        axes[0].axis('off')
        # Reconstructed Image
        axes[1].imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        axes[1].set_title("Reconstructed")
        axes[1].axis('off')
        plt.show()

def main():
    # ============================
    # 1. Configuration
    # ============================
    train_csv = "data/training.csv"          # Path to your CSV file with image IDs and labels for training
    train_image_dir = 'data/training/'       # Path to your training images
    val_csv = "data/validation.csv"          # Path to your CSV file with image IDs and labels for validation
    val_image_dir = 'data/validation/'       # Path to your validation images
    batch_size = 8
    latent_dim = 128                          # Increased from 50 to 128 for richer latent representations
    num_epochs_ae = 50
    num_epochs_clf = 100
    learning_rate_ae = 1e-3                    # Reduced from 1e-2 to 1e-3 for stable AE training
    learning_rate_clf = 1e-3
    patience = 10                              # For early stopping
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ============================
    # 2. Data Transforms
    # ============================
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),  # Scales images to [0,1]
    ])

    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # Scales images to [0,1]
    ])
    
    # ============================
    # 3. Load Datasets
    # ============================
    dataset_train = LoadTifDataset(csv_file=train_csv, image_dir=train_image_dir, transform=transform_train)
    dataset_val = LoadTifDataset(csv_file=val_csv, image_dir=val_image_dir, transform=transform_val)
    
    # ============================
    # 4. Handle Class Imbalance
    # ============================
    labels_train = [label for _, label in dataset_train]
    class_counts = np.bincount(labels_train)
    if len(class_counts) < 2:
        # Ensure there are at least two classes
        class_counts = np.pad(class_counts, (0, 2 - len(class_counts)), 'constant')
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # Normalize weights
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # ============================
    # 5. Initialize DataLoaders
    # ============================
    train_loader_ae = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_ae = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # ============================
    # 6. Initialize Autoencoder
    # ============================
    autoencoder = Autoencoder(latent_dim=latent_dim).to(device)
    criterion_ae = nn.MSELoss()
    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=learning_rate_ae)
    
    # Learning Rate Scheduler for AE (Optional)
    scheduler_ae = optim.lr_scheduler.StepLR(optimizer_ae, step_size=20, gamma=0.1)
    
    print("Starting Autoencoder Training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # ============================
    # 7. Training Loop for Autoencoder with Early Stopping
    # ============================
    for epoch in range(num_epochs_ae):
        autoencoder.train()
        running_loss = 0.0
        for images, _ in train_loader_ae:
            images = images.to(device)
            optimizer_ae.zero_grad()
            outputs = autoencoder(images)
            loss = criterion_ae(outputs, images)
            loss.backward()
            optimizer_ae.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader_ae.dataset)
        
        # Validation
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader_ae:
                images = images.to(device)
                outputs = autoencoder(images)
                loss = criterion_ae(outputs, images)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader_ae.dataset)
        
        # Print Losses
        print(f"Epoch [{epoch+1}/{num_epochs_ae}], Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best AE model
            torch.save(autoencoder.state_dict(), 'models/best_autoencoder.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered for Autoencoder!")
                break
        
        # Step the scheduler
        scheduler_ae.step()
        
        # Visualize reconstructions every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_reconstructions(autoencoder, val_loader_ae, device)
    
    # ============================
    # 8. Load the Best Autoencoder
    # ============================
    autoencoder.load_state_dict(torch.load('models/best_autoencoder.pth'))
    print("Autoencoder training completed and best model saved.")
    
    # ============================
    # 9. Feature Extraction
    # ============================
    autoencoder.eval()
    
    def extract_features(loader):
        features = []
        labels = []
        with torch.no_grad():
            for images, lbls in loader:
                images = images.to(device)
                encoded = autoencoder.encoder(images)             # (batch_size, 512, 1, 1)
                encoded = torch.flatten(encoded, 1)               # (batch_size, 512)
                latent = autoencoder.fc_enc(encoded)              # (batch_size, latent_dim)
                features.append(latent.cpu())
                labels.append(lbls)
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        return features, labels
    
    print("Extracting latent features...")
    train_features, train_labels = extract_features(train_loader_ae)
    val_features, val_labels = extract_features(val_loader_ae)
    
    # ============================
    # 10. Initialize Classifier
    # ============================
    classifier = Classifier(latent_dim=latent_dim, num_classes=2).to(device)
    criterion_clf = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer_clf = optim.Adam(classifier.parameters(), lr=learning_rate_clf, weight_decay=1e-5)  # Added weight_decay
    
    # Learning Rate Scheduler for Classifier (Optional)
    scheduler_clf = optim.lr_scheduler.StepLR(optimizer_clf, step_size=40, gamma=0.1)
    
    # ============================
    # 11. Create DataLoaders for Classifier
    # ============================
    train_dataset_clf = torch.utils.data.TensorDataset(train_features, train_labels)
    val_dataset_clf = torch.utils.data.TensorDataset(val_features, val_labels)
    
    train_loader_clf = DataLoader(train_dataset_clf, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_clf = DataLoader(val_dataset_clf, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Starting Classifier Training...")
    best_custom_score = 0
    epochs_no_improve_clf = 0
    
    # ============================
    # 12. Training Loop for Classifier with Early Stopping
    # ============================
    for epoch in range(num_epochs_clf):
        classifier.train()
        running_loss = 0.0
        for features, labels in train_loader_clf:
            features = features.to(device)
            labels = labels.to(device)
            optimizer_clf.zero_grad()
            outputs = classifier(features)
            loss = criterion_clf(outputs, labels)
            loss.backward()
            optimizer_clf.step()
            running_loss += loss.item() * features.size(0)
        epoch_loss = running_loss / len(train_loader_clf.dataset)
        
        # Validation
        classifier.eval()
        val_loss = 0.0
        a_0 = 0  # Correct predictions for class 0
        a_1 = 0  # Correct predictions for class 1
        n_0 = 0  # Total actual class 0
        n_1 = 0  # Total actual class 1
        with torch.no_grad():
            for features, labels in val_loader_clf:
                features = features.to(device)
                labels = labels.to(device)
                outputs = classifier(features)
                loss = criterion_clf(outputs, labels)
                val_loss += loss.item() * features.size(0)
                
                _, preds = torch.max(outputs, 1)
                
                # Update counts for custom score
                for pred, true in zip(preds, labels):
                    if true == 0:
                        n_0 += 1
                        if pred == 0:
                            a_0 += 1
                    elif true == 1:
                        n_1 += 1
                        if pred == 1:
                            a_1 += 1
        val_loss /= len(val_loader_clf.dataset)
        
        # Calculate custom score
        custom_score = calculate_custom_score(a_0, a_1, n_0, n_1)
        
        # Print metrics
        print(f"Epoch [{epoch+1}/{num_epochs_clf}], Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}, Custom Score: {custom_score:.6f}")
        
        # Step the scheduler
        scheduler_clf.step()
    
    # ============================
    # 13. Load the Best Classifier
    # ============================
    classifier.load_state_dict(torch.load('models/best_classifier.pth'))
    print("Classifier training completed and best model saved.")
    
    # ============================
    # 14. Final Evaluation on Validation Set
    # ============================
    print("Starting Final Evaluation on Validation Set...")
    classifier.eval()
    a_0_final = 0  # Correct predictions for class 0
    a_1_final = 0  # Correct predictions for class 1
    n_0_final = 0  # Total actual class 0
    n_1_final = 0  # Total actual class 1
    with torch.no_grad():
        for features, labels in val_loader_clf:
            features = features.to(device)
            labels = labels.to(device)
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            
            # Update counts for custom score
            for pred, true in zip(preds, labels):
                if true == 0:
                    n_0_final += 1
                    if pred == 0:
                        a_0_final += 1
                elif true == 1:
                    n_1_final += 1
                    if pred == 1:
                        a_1_final += 1
    final_custom_score = calculate_custom_score(a_0_final, a_1_final, n_0_final, n_1_final)
    
    print(f"Final Custom Score on Validation Set: {final_custom_score:.6f}")
    

if __name__ == "__main__":
    main()
