# predict_on_eval.py

import os
import json
import argparse
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import (
    get_transforms,
    get_model,
    load_model,
    preprocess_image,
    predict
)

def main():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    
    # Updated arguments with default values for validation
    parser.add_argument(
        '--image_dir',
        type=str,
        default='data/validation',  # Default path to the validation images folder
        help='Path to the directory containing evaluation images. Default is "validation".'
    )
    parser.add_argument(
        '--labels_csv',
        type=str,
        default='data/validation.csv',  # Default path to the validation labels CSV
        help='Path to the CSV file containing image labels. Default is "validation.csv".'
    )
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        default='checkpoints/final_model.pth',
        help='Path to the model checkpoint. Default is "checkpoints/final_model.pth".'
    )
    parser.add_argument(
        '--model_info',
        type=str,
        default='checkpoints/final_model_info.json',
        help='Path to the model info JSON. Default is "checkpoints/final_model_info.json".'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for classification. Default is 0.5.'
    )
    
    args = parser.parse_args()

    # Display the configuration being used
    print("Evaluation Configuration:")
    print(f"Image Directory : {args.image_dir}")
    print(f"Labels CSV      : {args.labels_csv}")
    print(f"Model Checkpoint: {args.model_checkpoint}")
    print(f"Model Info      : {args.model_info}")
    print(f"Threshold       : {args.threshold}\n")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    if not os.path.exists(args.model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")
    if not os.path.exists(args.model_info):
        raise FileNotFoundError(f"Model info file not found: {args.model_info}")

    model, img_size = load_model(args.model_checkpoint, args.model_info, device)
    print(f"Loaded model architecture: {model_info['model_name']} with image size: {img_size}\n")

    # Get transforms
    transform = get_transforms(img_size=img_size)

    # Load evaluation data
    if not os.path.exists(args.labels_csv):
        raise FileNotFoundError(f"Labels CSV file not found: {args.labels_csv}")
    
    df = pd.read_csv(args.labels_csv)
    
    # Validate CSV columns
    required_columns = {'image_id', 'is_homogenous'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file must contain the following columns: {required_columns}")
    
    image_ids = df['image_id'].tolist()
    labels = df['is_homogenous'].tolist()

    predictions = []
    ground_truths = []

    print("Starting predictions...\n")
    
    for idx, (img_id, label) in enumerate(zip(image_ids, labels), start=1):
        # Convert img_id to integer and format filename
        try:
            img_id_int = int(img_id)
        except ValueError:
            print(f"[{idx}/{len(image_ids)}] Invalid image_id '{img_id}'. Skipping.")
            continue

        # Format the image filename with leading zeros and .tif extension
        img_filename = f"{img_id_int:03d}.tif"  # Adjust '03d' if necessary
        full_image_path = os.path.join(args.image_dir, img_filename)
        
        if not os.path.exists(full_image_path):
            print(f"[{idx}/{len(image_ids)}] Image not found: {full_image_path}. Skipping.")
            continue
        try:
            image_tensor = preprocess_image(full_image_path, transform, device)
            prediction = predict(image_tensor, model, device, threshold=args.threshold)
            predictions.append(prediction)
            ground_truths.append(int(label))
            print(f"[{idx}/{len(image_ids)}] Processed: {img_filename} | Prediction: {'Homogeneous' if prediction == 1 else 'Heterogeneous'} | Ground Truth: {'Homogeneous' if label == 1 else 'Heterogeneous'}")
        except Exception as e:
            print(f"[{idx}/{len(image_ids)}] Error processing image {full_image_path}: {e}. Skipping.")
            continue

    # Check if any predictions were made
    if not predictions:
        print("No predictions were made. Please check your data and try again.")
        return

    # Compute evaluation metrics
    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, zero_division=0)
    recall = recall_score(ground_truths, predictions, zero_division=0)
    f1 = f1_score(ground_truths, predictions, zero_division=0)

    # Compute custom Score
    # a0: Correctly predicted label 0
    # a1: Correctly predicted label 1
    # n0: Total true label 0
    # n1: Total true label 1

    a0 = sum((gt == 0 and pred == 0) for gt, pred in zip(ground_truths, predictions))
    a1 = sum((gt == 1 and pred == 1) for gt, pred in zip(ground_truths, predictions))
    n0 = sum(gt == 0 for gt in ground_truths)
    n1 = sum(gt == 1 for gt in ground_truths)

    # Handle division by zero
    if n0 == 0 or n1 == 0:
        score = 0.0
        print("\nWarning: One of the classes has zero instances. Custom Score set to 0.0.")
    else:
        score = (a0 * a1) / (n0 * n1)

    print("\nEvaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Custom Score: {score:.4f}")

if __name__ == "__main__":
    main()
