# train.py

# Import necessary libraries
import os
import pandas as pd
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Define paths to your training CSV and image folder
csv_file_path = "data/training.csv"  # Change this to your training CSV file path
image_folder_path = "data/training/"  # Change this to your training image folder path

# Define the path where the model should be saved
model_save_path = "models/svm_model.pkl"

# Create the directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load the CSV file
csv_data = pd.read_csv(csv_file_path)

# Strip any extra whitespace in the column names (if applicable)
csv_data.columns = csv_data.columns.str.strip()

# Set the desired image size (e.g., 128x128) for resizing
image_size = (128, 128)

# Initialize lists to store images and labels
images = []
labels = []

# Iterate over the rows in the CSV to load images and their corresponding labels
for index, row in csv_data.iterrows():
    image_id = str(row['image_id']).zfill(3)  # Format image_id as a 3-digit string if needed
    label = row['is_homogenous']
    image_path = os.path.join(image_folder_path, f"{image_id}.tif")
    
    if os.path.exists(image_path):
        # Open, resize, and convert the image to a numpy array
        img = Image.open(image_path).resize(image_size)
        img_array = np.array(img)
        images.append(img_array)
        labels.append(label)

# Extract HOG features from each image
hog_features = []
for image in images:
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = Image.fromarray(image).convert('L')
        image = np.array(image)
    # Compute HOG features for the image
    features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    hog_features.append(features)

# Convert features and labels to numpy arrays
hog_features_np = np.array(hog_features)
labels_np = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features_np, labels_np, test_size=0.2, random_state=42)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Heterogeneous', 'Homogeneous'])

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Save the trained model to disk
joblib.dump(svm_classifier, model_save_path)
print(f"Model saved to {model_save_path}")
