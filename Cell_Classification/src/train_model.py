# train_model.py

# Import necessary libraries
import os
import pandas as pd
import numpy as np
from utils import load_sample  # Import load_sample from utils
import base64  # <-- Add this import
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

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

# Initialize lists to store HOG features and labels
hog_features = []
labels = []

# Iterate over the rows in the CSV to load images and their corresponding labels
for index, row in csv_data.iterrows():
    image_id = str(row['image_id']).zfill(3)  # Format image_id as a 3-digit string if needed
    label = row['is_homogenous']
    image_filename = f"{image_id}.tif"
    image_path = os.path.join(image_folder_path, image_filename)
    
    if os.path.exists(image_path):
        try:
            # Read the image file as bytes
            with open(image_path, "rb") as img_file:
                encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Use load_sample to decode and load the image
            sample = load_sample(encoded_img)
            img_array = sample["image"]
            
            # Append the image and label
            hog_features.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error processing image '{image_filename}': {e}")
    else:
        print(f"Image file '{image_filename}' not found in '{image_folder_path}'.")

# Now, extract HOG features using preprocess_image from utils
from utils import preprocess_image

processed_hog_features = []
for idx, image in enumerate(hog_features):
    try:
        features = preprocess_image(image)
        processed_hog_features.append(features.flatten())
    except Exception as e:
        print(f"Error extracting HOG features for image index {idx}: {e}")

# Convert features and labels to numpy arrays
hog_features_np = np.array(processed_hog_features)
labels_np = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    hog_features_np, labels_np, test_size=0.2, random_state=42
)

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
