# train.py

# Import necessary libraries
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import joblib

# Define paths to your training CSV and image folder
csv_file_path = "data/training.csv"  # Change this to your training CSV file path
image_folder_path = "data/training/"  # Change this to your training image folder path

# Define the path where the model should be saved
model_save_path = "models/svm_model_optimized.pkl"

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

# Convert features and labels to numpy arrays
images_np = np.array(images)
labels_np = np.array(labels)

# Balance the dataset by oversampling the minority class (homogeneous)
images_homogeneous = images_np[labels_np == 1]
labels_homogeneous = labels_np[labels_np == 1]
images_heterogeneous = images_np[labels_np == 0]
labels_heterogeneous = labels_np[labels_np == 0]

# Oversample the homogeneous class
images_homogeneous_resampled, labels_homogeneous_resampled = resample(
    images_homogeneous, labels_homogeneous,
    replace=True,  # Oversample
    n_samples=len(images_heterogeneous),  # Match the number of heterogeneous samples
    random_state=42
)

# Combine back to a balanced dataset
images_balanced = np.vstack((images_heterogeneous, images_homogeneous_resampled))
labels_balanced = np.hstack((labels_heterogeneous, labels_homogeneous_resampled))

# Extract HOG features from each image
hog_features = []
for image in images_balanced:
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = Image.fromarray(image).convert('L')
        image = np.array(image)
    # Compute HOG features for the image
    features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    hog_features.append(features)

# Convert HOG features to numpy arrays
hog_features_np = np.array(hog_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features_np, labels_balanced, test_size=0.2, random_state=42)

# Train an SVM classifier with hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(SVC(random_state=42), param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best estimator
best_svm = grid_search.best_estimator_

# Predict on the test set using the best model
y_pred = best_svm.predict(X_test)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Heterogeneous', 'Homogeneous'])

# Print the results
print(f"Best Model Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Save the trained model to disk
joblib.dump(best_svm, model_save_path)
print(f"Model saved to {model_save_path}")
