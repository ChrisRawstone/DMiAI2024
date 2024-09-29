import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from models.model import SimpleClassifier
from predict_model import predict_local
from data.make_dataset import LoadTifDataset, PadToSize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'I am on the device: {device}')

def calculate_custom_score(a_0, a_1, n_0, n_1):
    # Ensure no division by zero
    if n_0 == 0 or n_1 == 0:
        return 0
    return (a_0 * a_1) / (n_0 * n_1)

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, num_epochs=5):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

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

            # Update running loss
            running_loss += loss.item() * inputs.size(0)

        # Step the scheduler
        scheduler.step()

        epoch_loss = running_loss / len(train_dataloader.dataset)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_inputs, val_labels in val_dataloader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item() * val_inputs.size(0)

            val_epoch_loss = val_loss / len(val_dataloader.dataset)

            # Calculate custom score on validation set
            score_val = predict_local(model, val_dataloader, calculate_custom_score, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, "
              f"Val Custom Score: {score_val:.4f}")

    print("Training complete!")
    return model

final_resize_size = (224, 224)

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.Resize(final_resize_size),  # Resize the shorter side first    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.Resize(final_resize_size),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize as per ImageNet
])

# Create datasets
csv_file = 'data/training.csv'
image_dir = 'data/training/'

csv_file_val = 'data/validation.csv'
image_dir_val = 'data/validation/'

csv_file_aug = 'data/augmented_training.csv'
image_dir_aug = 'data/train_augmentation/'

train_dataset = LoadTifDataset(csv_file=csv_file, image_dir=image_dir, transform=train_transform)
train_aug_dataset = LoadTifDataset(csv_file=csv_file_aug, image_dir=image_dir_aug, transform=train_transform)
val_dataset = LoadTifDataset(csv_file=csv_file_val, image_dir=image_dir_val, transform=val_transform)

combined_train_dataset = ConcatDataset([train_dataset, train_aug_dataset])
train_dataloader = DataLoader(combined_train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

model = SimpleClassifier().to(device) 
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

num_epochs = 200
model = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, num_epochs=num_epochs)

torch.save(model.state_dict(), 'trained_model_cell.pth')

# Optionally, if you still want to compute scores after training
score_train = predict_local(model, train_dataloader, calculate_custom_score, device)
print(f"Final Training Custom Score: {score_train:.4f}")

# Validation score is already computed each epoch; if you want the final one:
score_val = predict_local(model, val_dataloader, calculate_custom_score, device)
print(f"Final Validation Custom Score: {score_val:.4f}")
