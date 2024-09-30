# utils.py

import os
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models

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
        nn.Module: The loaded model.
        int: Image size used by the model.
    """
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    model_name = model_info['model_name']
    img_size = model_info['img_size']

    model = get_model(model_name, num_classes=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, img_size

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

def predict_batch(image_tensors, model, device, threshold=0.5):
    """
    Predicts classes for a batch of images.

    Args:
        image_tensors (torch.Tensor): Batch of preprocessed image tensors.
        model (nn.Module): Loaded model.
        device (torch.device): Device to perform computation on.
        threshold (float): Threshold for binary classification.

    Returns:
        List[int]: Predictions for each image in the batch.
    """
    with torch.no_grad():
        outputs = model(image_tensors)
        probs = torch.sigmoid(outputs)
        predictions = (probs > threshold).int().squeeze().tolist()
        if isinstance(predictions, int):
            predictions = [predictions]
    return predictions




import os
import numpy as np
import cv2
import base64 
from PIL import Image
from skimage.feature import hog
import joblib

# Set the desired image size (e.g., 128x128) for resizing
IMAGE_SIZE = (128, 128)

def decode_image(encoded_img: str) -> np.ndarray:
    """
    Decodes a base64 encoded image string to a NumPy array.

    Args:
        encoded_img (str): Base64 encoded image string.

    Returns:
        np.ndarray: Decoded image.
    """
    try:
        # Decode the base64 string to bytes
        img_data = base64.b64decode(encoded_img)
        # Convert bytes data to NumPy array
        np_arr = np.frombuffer(img_data, np.uint8)
        # Decode the image data using OpenCV
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image decoding resulted in None.")
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

def load_sample(encoded_img: str) -> dict:
    """
    Loads and decodes the sample image.

    Args:
        encoded_img (str): Base64 encoded image string.

    Returns:
        dict: Dictionary containing the image.
    """
    image = decode_image(encoded_img)  # Decode the image
    return {
        "image": image
    }

def load_model(model_path: str):
    """
    Load the trained model from the given path.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        model: The loaded machine learning model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image for prediction.
    This includes converting to grayscale, resizing, and extracting HOG features.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: Extracted HOG features from the image.
    """
    # Convert to grayscale if image has multiple channels
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image
    image_resized = cv2.resize(image, IMAGE_SIZE)

    # Compute HOG features
    hog_features = hog(
        image_resized, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        visualize=False,
        feature_vector=True
    )

    return hog_features.reshape(1, -1)  # Reshape to match the input format expected by the model




def save_image_as_tif(image: np.ndarray, output_path: str) -> None:
    """
    Saves the provided image as a .tif file in 'I;16B' format.

    Args:
        image (np.ndarray): Image to be saved.
        output_path (str): File path where the image should be saved.

    Raises:
        ValueError: If the image is not a valid NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image is not a valid NumPy array.")

    # Convert to PIL Image
    # Ensure image is in grayscale before conversion
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    # Convert NumPy array to PIL Image with mode 'I;16'
    pil_image = Image.fromarray(image_gray.astype(np.uint16), mode='I;16')

    # Save as .tif in 'I;16B' format
    pil_image.save(output_path, format='TIFF')