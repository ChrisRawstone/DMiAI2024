# predict_on_eval.py

import os
import pandas as pd
import numpy as np
from src.utils import load_model, load_sample  # Import necessary functions from utils
from src.predict import predict
import base64
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

# Define paths
model_path = "models/svm_model.pkl"  # Path to the saved model
validation_csv_path = "data/validation.csv"  # CSV containing filenames and labels
validation_dir = "data/validation"  # Directory containing validation images

# Load the trained model
model = load_model(model_path)

# Check if the validation CSV exists
if not os.path.exists(validation_csv_path):
    print(f"Validation CSV '{validation_csv_path}' does not exist.")
    exit(1)

# Load the validation CSV file
validation_data = pd.read_csv(validation_csv_path)

# Ensure that the 'image_id' column is of string type (remove leading/trailing spaces in column names if any)
validation_data.columns = validation_data.columns.str.strip()
validation_data['image_id'] = validation_data['image_id'].astype(str)

# Format image_id as a 3-digit string if needed and add the file extension
validation_data['image_id'] = validation_data['image_id'].str.zfill(3) + ".tif"

# Extract filenames and corresponding labels
file_label_dict = dict(zip(validation_data['image_id'], validation_data['is_homogenous']))

# Initialize counters for class-wise metrics
n_0, n_1 = 0, 0  # Total samples for each class
a_0, a_1 = 0, 0  # Correctly predicted samples for each class

# Lists to hold ground truth and predicted labels for calculating metrics
true_labels = []
predicted_labels = []

# Load the trained model
model_path = "models/svm_model.pkl"  # Path to the saved model
model = joblib.load(model_path)
print(f"Model loaded from {model_path}")

# Check if the validation directory exists
if os.path.exists(validation_dir):
    # Iterate through each image file listed in the validation CSV
    for filename, true_label in file_label_dict.items():
        # Construct the full path to the image
        image_path = os.path.join(validation_dir, filename)

        # Check if the image file exists in the directory
        if os.path.isfile(image_path):
            try:
                # Read the image file as bytes
                with open(image_path, "rb") as img_file:
                    encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Use load_sample to decode and load the image
                sample = load_sample(encoded_img)
                image = sample["image"]

                # Predict using the loaded model
                prediction = predict(image)

                # Append to the lists for metrics calculation
                true_labels.append(true_label)
                predicted_labels.append(prediction)

                # Increment total counts for each class
                if true_label == 0:
                    n_0 += 1
                    if prediction == 0:
                        a_0 += 1
                elif true_label == 1:
                    n_1 += 1
                    if prediction == 1:
                        a_1 += 1
            except Exception as e:
                print(f"Error processing image '{filename}': {e}")
        else:
            print(f"Image file '{filename}' listed in '{validation_csv_path}' not found in directory '{validation_dir}'.")

    # Calculate precision, recall, and F1 score
    precision = precision_score(true_labels, predicted_labels, average='binary', pos_label=1)
    recall = recall_score(true_labels, predicted_labels, average='binary', pos_label=1)
    f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)

    # Calculate custom score using the provided formula
    if n_0 > 0 and n_1 > 0:
        custom_score = (a_0 * a_1) / (n_0 * n_1)
    else:
        custom_score = 0  # Avoid division by zero

    # Print out the metrics
    print(f"Total samples from class 0 (Heterogeneous): {n_0}")
    print(f"Total samples from class 1 (Homogeneous): {n_1}")
    print(f"Correctly predicted samples from class 0 (Heterogeneous): {a_0}")
    print(f"Correctly predicted samples from class 1 (Homogeneous): {a_1}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Custom Score: {custom_score:.4f}")
else:
    print(f"Validation directory '{validation_dir}' does not exist.")
