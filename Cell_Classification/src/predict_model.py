import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from collections import Counter
import numpy as np
from models.model import SimpleClassifier
from data.make_dataset import LoadTifDataset, PadToSize
import torch.nn.functional as F

def predict_local(model, data_loader, calculate_custom_score, device) -> int:
    model.eval()
    total_0 = 0
    total_1 = 0
    correct_0 = 0
    correct_1 = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            probabilities = F.softmax(logits, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
                    
            # Calculate correct predictions and totals for each class
            total_0 += (labels == 0).sum().item()
            total_1 += (labels == 1).sum().item()

            correct_0 += ((predicted == 0) & (labels == 0)).sum().item()
            correct_1 += ((predicted == 1) & (labels == 1)).sum().item()
            
            # print(f'labels:{labels}')
            # print(f'predicted:{predicted}')

    # Calculate the custom score
    custom_score = calculate_custom_score(correct_0, correct_1, total_0, total_1)
    
    return custom_score

# def calculate_custom_score(a_0, a_1, n_0, n_1):
#     # Ensure no division by zero
#     if n_0 == 0 or n_1 == 0:
#         return 0
#     return (a_0 * a_1) / (n_0 * n_1)

# # Define the padding size
# target_width = 1500
# target_height = 1470

# # Define transformations for validation data (no augmentation)
# val_transform = transforms.Compose([
#     PadToSize(target_width=target_width, target_height=target_height),
#     # transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet mean and std

# ])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SimpleClassifier()
# model.load_state_dict(torch.load('/dtu/blackhole/00/156512/DM-i-AI-2024/Cell_Classification/src/trained_model_cell.pth'))
# model = model.to(device)  # Move model to the correct device

# image_dir = "data/validation"
# csv_file_path = "data/validation.csv"
# val_dataset = LoadTifDataset(image_dir=image_dir, csv_file_path=csv_file_path, transform=val_transform)
# val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)
# score_val = predict_local(model, val_dataloader, calculate_custom_score, device)
# print(score_val)