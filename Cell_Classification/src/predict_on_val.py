# predict_on_val.py
import argparse
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.data.make_dataset import LoadTifDataset

from utils import (
    get_transforms,
    load_model,
    preprocess_image,
    predict,
    calculate_custom_score
)

def main():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    
    # Updated arguments with default values for validation
    parser.add_argument(
        '--image_dir',
        type=str,
        default='data/validation16bit',  # Default path to the validation images folder
        help='Path to the directory containing evaluation images. Default is "data/validation".'
    )
    parser.add_argument(
        '--labels_csv',
        type=str,
        default='data/validation.csv',  # Default path to the validation labels CSV
        help='Path to the CSV file containing image labels. Default is "data/validation.csv".'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for classification. Default is 0.5.'
    )
    
    args = parser.parse_args()

    model_checkpoint = 'checkpoints/best_model_optuna.pth'
    model_info = 'checkpoints/test.json'
    # model_checkpoint = 'checkpoints/best_model_2_0.263.pth'
    # model_info = 'checkpoints/best_model_2_0.263.json'


    # Display the configuration being used
    print("Evaluation Configuration:")
    print(f"Image Directory : {args.image_dir}")
    print(f"Labels CSV      : {args.labels_csv}")
    print(f"Model Checkpoint: {model_checkpoint}")
    print(f"Model Info      : {model_info}")
    print(f"Threshold       : {args.threshold}\n")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    model, img_size, model_info = load_model(model_checkpoint, model_info, device)
    print(f"Loaded model architecture: {model_info['model_name']} with image size: {img_size}\n")

    # Load the validation dataset
    # image_dir_val = args.image_dir
    # csv_file_path_val = args.labels_csv
    
    # image_dir_val = "data/test_val"
    # csv_file_path_val = "data/test_val.csv"
    
    image_dir_val = "data/validation"
    csv_file_path_val = "data/validation.csv"


    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.485, 0.485),  # Using ImageNet means
                    std=(0.229, 0.229, 0.229)),   # Using ImageNet stds
        ToTensorV2(),
    ])

    val_dataset = LoadTifDataset(image_dir=image_dir_val, csv_file_path=csv_file_path_val, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()

    val_preds = []
    val_targets = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):  # Updated autocast usage
                outputs = model(images)

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.detach().cpu().numpy())

    # Calculate custom score
    preds_binary = (np.array(val_preds) > 0.5).astype(int)
    custom_score = calculate_custom_score(val_targets, preds_binary)

    ground_truths = np.array(val_targets)
    predictions = np.array(preds_binary)

    # Compute evaluation metrics
    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, zero_division=0)
    recall = recall_score(ground_truths, predictions, zero_division=0)
    f1 = f1_score(ground_truths, predictions, zero_division=0)

    print("\nEvaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Custom Score: {custom_score:.4f}")

if __name__ == "__main__":
    main()