import numpy as np
import random
import torch
import os
from pathlib import Path
from PIL import Image, ImageOps
from torchvision import transforms
from src.models.model import SimpleClassifier
from src.data.make_dataset import PadToSize
from src.utils import tif_to_ndarray
import torch.nn.functional as F

# Get the script's directory
script_dir = Path(__file__).resolve()
# Go two levels up from the script's directory
parent_parent_dir = script_dir.parent
# Change the current working directory to the parent-parent directory
os.chdir(parent_parent_dir)

# ### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
# def predict(image: np.ndarray) -> int:

#     is_homogenous = example_model(image)

#     return is_homogenous


# def example_model(image) -> int:

#     is_homogenous = random.randint(0, 1)

#     return is_homogenous



def predict_local() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load('/home/azureuser/DM-i-AI-2024/Cell_Classification/model.pth', map_location=device, weights_only=True))
    model.eval()

    target_width = 1500
    target_height = 1470
    transform = transforms.Compose([
                                    PadToSize(target_width=target_width, target_height=target_height),  # Custom padding transform
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor()])
    
    image_dir = "data/training"
    img_path = os.path.join(image_dir, "001.tif")
    image = tif_to_ndarray(img_path)
    image = Image.fromarray(image)
    image = transform(image)
    # Move image tensor to the same device as the model
    image = image.unsqueeze(0).to(device)

    logits = model(image)
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)
    # Get the predicted class (the index of the highest probability)
    predicted_class = torch.argmax(probabilities, dim=1)

    return predicted_class

target_width = 1500
target_height = 1470
transform = transforms.Compose([
                                PadToSize(target_width=target_width, target_height=target_height),  # Custom padding transform
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleClassifier(num_classes=2).to(device)
model.load_state_dict(torch.load('/home/azureuser/DM-i-AI-2024/Cell_Classification/model.pth', map_location=device, weights_only=True))
model.eval()

def predict(image: np.ndarray) -> int:
    image = image['image']
    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    logits = model(image)
    probabilities = F.softmax(logits, dim=1)
    # Get the predicted class (the index of the highest probability)
    is_homogenous = torch.argmax(probabilities, dim=1)
    is_homogenous = is_homogenous.item()
    
    print("predicting: ",)
    return is_homogenous


# def test_predict():
#     image_dir = "data/training"
#     img_path = os.path.join(image_dir, "001.tif")
#     image = tif_to_ndarray(img_path)
#     is_homogenous = predict(image)
#     #is_homogenous = is_homogenous.item()
#     return is_homogenous

# is_homogenous=test_predict()