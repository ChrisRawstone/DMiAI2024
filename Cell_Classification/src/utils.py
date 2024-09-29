# utils.py

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