# predict.py

# Import necessary libraries
import os
import pandas as pd
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Define paths for the validation data and model
validation_csv_path = "data/validation.csv"  # Change this to your validation CSV path
validation_image_folder_path = "data/validation/"  # Change this to your validation image folder path
model_load_path = "models/svm_model_optimized.pkl"  # Path to the saved model

# Load the validation CSV file
validation_data = pd.read_csv(validation_csv_path)

# Strip any extra whitespace in the column names (if applicable)
validation_data.columns = validation_data.columns.str.strip()

# Set the desired image size (e.g., 128x128) for resizing
image_size = (128, 128)

# Load the saved model
svm_classifier = joblib.load(model_load_path)
print(f"Model loaded from {model_load_path}")

# Initialize lists to store images and labels
validation_images = []
validation_labels = []

# Iterate over the rows in the CSV to load images and their corresponding labels
for index, row in validation_data.iterrows():
    image_id = str(row['image_id']).zfill(3)  # Format image_id as a 3-digit string if needed
    label = row['is_homogenous']
    image_path = os.path.join(validation_image_folder_path, f"{image_id}.tif")
    
    if os.path.exists(image_path):
        # Open, resize, and convert the image to a numpy array
        img = Image.open(image_path).resize(image_size)
        img_array = np.array(img)
        validation_images.append(img_array)
        validation_labels.append(label)

# Extract HOG features from each validation image
validation_hog_features = []
for image in validation_images:
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = Image.fromarray(image).convert('L')
        image = np.array(image)
    # Compute HOG features for the image
    features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    validation_hog_features.append(features)

# Convert features and labels to numpy arrays
validation_hog_features_np = np.array(validation_hog_features)
validation_labels_np = np.array(validation_labels)

# Predict on the validation set
y_validation_pred = svm_classifier.predict(validation_hog_features_np)

# Calculate accuracy and classification report for validation set
validation_accuracy = accuracy_score(validation_labels_np, y_validation_pred)
validation_report = classification_report(validation_labels_np, y_validation_pred, target_names=['Heterogeneous', 'Homogeneous'])

# Print validation results
print(f"Validation Accuracy: {validation_accuracy:.2f}")
print("\nValidation Classification Report:")
print(validation_report)
