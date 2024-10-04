# predict.py

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import cv2
import pandas as pd 

def get_transforms(img_size=224):
    """
    Returns the transform pipeline used during training.

    Args:
        img_size (int): Image size for resizing.

    Returns:
        albumentations.Compose: Composed transformations.
    """
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),  # Using ImageNet means
                    std=(0.229, 0.224, 0.225)),   # Using ImageNet stds
        ToTensorV2(),
    ])
    return transform

def get_model(model_name, num_classes=1):
    """
    Constructs the model based on the architecture name.

    Args:
        model_name (str): Name of the model architecture.
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: The constructed model.
    """
    if model_name == 'ViT':
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        vit_weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=vit_weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    elif model_name == 'ResNet50':
        from torchvision.models import resnet50, ResNet50_Weights
        resnet_weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=resnet_weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'EfficientNet':
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        effnet_weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=effnet_weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'MobileNetV3':
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        mobilenet_weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=mobilenet_weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    return model

def load_model(checkpoint_path, model_info_path, device):
    """
    Loads the model architecture and weights.

    Args:
        checkpoint_path (str): Path to the model weights.
        model_info_path (str): Path to the model architecture info JSON.
        device (torch.device): Device to load the model on.

    Returns:
        tuple: (model, img_size, model_info)
    """
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    model_name = model_info['model_name']
    img_size = model_info['img_size']

    model = get_model(model_name, num_classes=1)
    
    # Set weights_only=True to address FutureWarning
    # Ensure your checkpoint is compatible with weights_only=True
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, img_size, model_info

def preprocess_image(image_path, transform, device):
    """
    Preprocesses the input image.

    Args:
        image_path (str): Path to the input image.
        transform (albumentations.Compose): Transform pipeline.
        device (torch.device): Device to load the image on.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Handle 16-bit images by converting to 8-bit
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)

    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # Handle images with alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transformations
    augmented = transform(image=image)
    image = augmented['image'].unsqueeze(0)  # Add batch dimension
    return image.to(device)

def predict(image_tensor, model, device, threshold=0.5):
    """
    Predicts the class of the input image.

    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor.
        model (nn.Module): Loaded model.
        device (torch.device): Device to perform computation on.
        threshold (float): Threshold for binary classification.

    Returns:
        int: 1 if homogeneous, 0 otherwise.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs)
        prediction = (probs > threshold).int().item()
    return prediction

def main():
    parser = argparse.ArgumentParser(description="Image Classification Prediction Script")
    parser.add_argument('--image', type=str, default='data/training/001.tif')
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints/final_model.pth', help='Path to the model checkpoint.')
    parser.add_argument('--model_info', type=str, default='checkpoints/final_model_info.json', help='Path to the model info JSON.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification.')
    args = parser.parse_args()

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    if not os.path.exists(args.model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")
    if not os.path.exists(args.model_info):
        raise FileNotFoundError(f"Model info file not found: {args.model_info}")

    model, img_size, model_info = load_model(args.model_checkpoint, args.model_info, device)

    print(f"Loaded model architecture: {model_info['model_name']} with image size: {img_size}")

    # Get transforms
    transform = get_transforms(img_size=img_size)

    # Preprocess image
    image_tensor = preprocess_image(args.image, transform, device)

    # Predict
    prediction = predict(image_tensor, model, device, threshold=args.threshold)
    print(f"Prediction for the image: {'Homogeneous' if prediction == 1 else 'Heterogeneous'}")

if __name__ == "__main__":
    main()
