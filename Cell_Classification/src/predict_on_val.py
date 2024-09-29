# evaluation.py

import os
import json
import argparse
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from predict import (
    get_transforms,
    get_model,
    load_model,
    preprocess_image,
    predict
)

def main():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing evaluation images.')
    parser.add_argument('--labels_csv', type=str, required=True, help='Path to the CSV file containing image labels.')
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints/final_model.pth', help='Path to the model checkpoint.')
    parser.add_argument('--model_info', type=str, default='checkpoints/final_model_info.json', help='Path to the model info JSON.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification.')
    args = parser.parse_args()

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    if not os.path.exists(args.model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")
    if not os.path.exists(args.model_info):
        raise FileNotFoundError(f"Model info file not found: {args.model_info}")

    model, img_size = load_model(args.model_checkpoint, args.model_info, device)

    # Get transforms
    transform = get_transforms(img_size=img_size)

    # Load evaluation data
    df = pd.read_csv(args.labels_csv)
    image_paths = df['image_path'].tolist()
    labels = df['label'].tolist()

    predictions = []
    ground_truths = []

    for img_path, label in zip(image_paths, labels):
        full_image_path = os.path.join(args.image_dir, img_path)
        if not os.path.exists(full_image_path):
            print(f"Image not found: {full_image_path}")
            continue
        try:
            image_tensor = preprocess_image(full_image_path, transform, device)
            prediction = predict(image_tensor, model, device, threshold=args.threshold)
            predictions.append(prediction)
            ground_truths.append(int(label))
        except Exception as e:
            print(f"Error processing image {full_image_path}: {e}")
            continue

    # Compute evaluation metrics
    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions)
    recall = recall_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions)

    print("\nEvaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

if __name__ == "__main__":
    main()
