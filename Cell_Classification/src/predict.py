# predict.py

# Import necessary libraries
import os
import numpy as np
from PIL import Image
from skimage.feature import hog
import joblib

# Load the trained model
model_path = "models/svm_model_optimized.pkl"  # Path to the saved model
model = joblib.load(model_path)
print(f"Model loaded from {model_path}")

# Set the desired image size (e.g., 128x128) for resizing
image_size = (128, 128)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image for prediction.
    This includes resizing and extracting HOG features.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: Extracted HOG features from the image.
    """
    # Convert to grayscale if image has multiple channels
    if len(image.shape) == 3:
        image = Image.fromarray(image).convert('L')
        image = np.array(image)

    # Resize the image
    image_resized = Image.fromarray(image).resize(image_size)
    image_resized = np.array(image_resized)

    # Compute HOG features
    hog_features = hog(image_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    return hog_features.reshape(1, -1)  # Reshape to match the input format expected by the model

def predict(image: np.ndarray) -> int:
    """
    Wrapper function for model prediction.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        int: 1 if homogenous, 0 otherwise.
    """
    # Preprocess the input image
    features = preprocess_image(image)

    # Predict using the model
    prediction = model.predict(features)

    # Return the prediction result as an integer
    return int(prediction[0])

# Example usage (if running the script directly):
if __name__ == "__main__":
    # Load a sample image for testing
    sample_image_path = "data/validation/001.tif"  # Replace with a sample image path
    if os.path.exists(sample_image_path):
        sample_image = np.array(Image.open(sample_image_path))
        # Predict using the sample image
        result = predict(sample_image)
        print(f"Prediction for the sample image: {'Homogeneous' if result == 1 else 'Heterogeneous'}")
