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
from PIL import Image
from skimage.feature import hog
import random

# Set the desired image size (e.g., 128x128) for resizing
IMAGE_SIZE = (224, 224)

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_custom_score(y_true, y_pred):
    """
    Calculates the custom score based on the formula:
    Score = (a0 * a1) / (n0 * n1)

    Where:
    - a0: True Negatives (correctly predicted as 0)
    - a1: True Positives (correctly predicted as 1)
    - n0: Total actual class 0 samples
    - n1: Total actual class 1 samples
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True Negatives (a0)
    a0 = np.sum((y_true == 0) & (y_pred == 0))

    # True Positives (a1)
    a1 = np.sum((y_true == 1) & (y_pred == 1))

    # Total actual class 0 and class 1
    n0 = np.sum(y_true == 0)
    n1 = np.sum(y_true == 1)

    # Avoid division by zero
    if n0 == 0 or n1 == 0:
        return 0.0

    score = (a0 * a1) / (n0 * n1)
    return score

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

# utils.py or wherever your get_model function resides
import torch.nn as nn
from torchvision import models
from timm import create_model  # Ensure you have `timm` installed for Swin Transformer models

def get_model(model_name, num_classes=1, pretrained=True, freeze=True):
    """
    Constructs the model based on the architecture name, modifies the final layer,
    and optionally freezes the pre-trained layers.

    Args:
        model_name (str): Name of the model architecture.
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pre-trained weights.
        freeze (bool): Whether to freeze the pre-trained layers.

    Returns:
        nn.Module: The constructed and configured model.
    """
    if model_name == 'ViT':
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        vit_weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=vit_weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    elif model_name == 'ViT32':
        from torchvision.models import vit_b_32, ViT_B_32_Weights
        vit_weights = ViT_B_32_Weights.DEFAULT if pretrained else None
        model = vit_b_32(weights=vit_weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    elif model_name == 'ResNet101':
        from torchvision.models import resnet101, ResNet101_Weights
        resnet_weights = ResNet101_Weights.DEFAULT if pretrained else None
        model = resnet101(weights=resnet_weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'ResNet50':
        from torchvision.models import resnet50, ResNet50_Weights
        resnet_weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=resnet_weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'EfficientNetB0':
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        effnet_weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=effnet_weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == 'EfficientNetB4':
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        effnet_weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = efficientnet_b4(weights=effnet_weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == 'MobileNetV3':
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        mobilenet_weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=mobilenet_weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name == 'DenseNet121':
        from torchvision.models import densenet121, DenseNet121_Weights
        densenet_weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=densenet_weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif model_name == 'SwinTransformer_B256':
        model = create_model('swinv2_base_window16_256', pretrained=pretrained, num_classes=num_classes)

    elif model_name == 'SwinTransformer_B224':
        model = create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=num_classes)

    elif model_name == 'SwinTransformer_H384':
        model = create_model('swin_large_patch4_window12_384', pretrained=pretrained, num_classes=num_classes)

    elif model_name == 'SwinTransformer_L384':
        model = create_model('swinv2_large_window12to24_192to384', pretrained=pretrained, num_classes=num_classes)

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the final layer(s)
        if model_name.startswith('ViT') or 'SwinTransformer' in model_name:
            # Assuming the last layer is named 'heads.head' or similar
            if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
                for param in model.heads.head.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            else:
                raise AttributeError("Cannot find the classifier head to unfreeze.")
        else:
            # For models like ResNet, DenseNet, EfficientNet, MobileNet
            if hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            else:
                raise AttributeError("Cannot find the classifier head to unfreeze.")

    return model


# def get_model(model_name, num_classes=1):
#     """
#     Constructs the model based on the architecture name.

#     Args:
#         model_name (str): Name of the model architecture.
#         num_classes (int): Number of output classes.

#     Returns:
#         nn.Module: The constructed model.
#     """
#     if model_name == 'ViT':
#         from torchvision.models import vit_b_16, ViT_B_16_Weights
#         vit_weights = ViT_B_16_Weights.DEFAULT
#         model = vit_b_16(weights=vit_weights)
#         in_features = model.heads.head.in_features
#         model.heads.head = nn.Linear(in_features, num_classes)
#     elif model_name == 'ViT32':
#         from torchvision.models import vit_b_32, ViT_B_32_Weights
#         vit_weights = ViT_B_32_Weights.DEFAULT
#         model = vit_b_32(weights=vit_weights)
#         in_features = model.heads.head.in_features
#         model.heads.head = nn.Linear(in_features, num_classes)
#     elif model_name == 'ResNet101':
#         from torchvision.models import resnet101, ResNet101_Weights
#         resnet_weights = ResNet101_Weights.DEFAULT
#         model = resnet101(weights=resnet_weights)
#         in_features = model.fc.in_features
#         model.fc = nn.Linear(in_features, num_classes)
#     elif model_name == 'ResNet50':
#         from torchvision.models import resnet50, ResNet50_Weights
#         resnet_weights = ResNet50_Weights.DEFAULT
#         model = resnet50(weights=resnet_weights)
#         in_features = model.fc.in_features
#         model.fc = nn.Linear(in_features, num_classes)
#     elif model_name == 'EfficientNetB0':
#         from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
#         effnet_weights = EfficientNet_B0_Weights.DEFAULT
#         model = efficientnet_b0(weights=effnet_weights)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, num_classes)
#     elif model_name == 'EfficientNetB4':
#         from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
#         effnet_weights = EfficientNet_B4_Weights.DEFAULT
#         model = efficientnet_b4(weights=effnet_weights)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, num_classes)
#     elif model_name == 'MobileNetV3':
#         from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
#         mobilenet_weights = MobileNet_V3_Large_Weights.DEFAULT
#         model = mobilenet_v3_large(weights=mobilenet_weights)
#         in_features = model.classifier[3].in_features
#         model.classifier[3] = nn.Linear(in_features, num_classes)
#     elif model_name == 'DenseNet121':
#         from torchvision.models import densenet121, DenseNet121_Weights
#         densenet_weights = DenseNet121_Weights.DEFAULT
#         model = densenet121(weights=densenet_weights)
#         in_features = model.classifier.in_features
#         model.classifier = nn.Linear(in_features, num_classes)
#     else:
#         raise ValueError(f"Unsupported model architecture: {model_name}")
#     return model

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
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
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

def preprocess_image_hog(image: np.ndarray) -> np.ndarray:
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