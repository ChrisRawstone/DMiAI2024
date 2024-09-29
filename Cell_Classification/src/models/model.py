# model.py

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture
class ResNet50BinaryClassifier(nn.Module):
    def __init__(self, model_path):
        super(ResNet50BinaryClassifier, self).__init__()
        # Load pre-trained ResNet50 model
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        # Modify the final layer for binary classification
        self.model.fc = nn.Linear(num_features, 1)
        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode

        # Define the same transformations used during training/evaluation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, image: np.ndarray) -> int:
        """
        Predicts whether the input cell image is homogenous or not.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            int: 1 if homogenous, 0 otherwise.
        """
        # Convert the NumPy array to PIL Image
        if image.ndim == 2:
            # Grayscale to RGB
            image = Image.fromarray(image).convert("RGB")
        elif image.ndim == 3:
            # Ensure it's in RGB format
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unsupported image shape: {}".format(image.shape))
        
        # Apply transformations
        input_tensor = self.transform(image).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output.squeeze()).item()
            prediction = int(prob >= 0.5)  # Threshold at 0.5

        return prediction

# Initialize the model once when the module is imported
MODEL_PATH = "resnet50_homogeneity_detection.pth"  # Ensure this path is correct
model = ResNet50BinaryClassifier(MODEL_PATH)


