import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from data.make_dataset import get_dataloaders_final_train
from src.utils import calculate_custom_score
from utils import get_model
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as pyplot

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model_info_dict = {
#     'MODELS_FINAL_EVAL/best_trained_model_1.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
#     'MODELS_FINAL_EVAL/best_trained_model_2.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
#     'MODELS_FINAL_EVAL/best_trained_model_3.pth': {'architecture': 'EfficientNetB0', 'img_size': 1000, 'batch_size': 4},
#     # 'MODELS_FINAL_EVAL/best_trained_model_4_1.pth': {'architecture': 'DenseNet121', 'img_size': 1000, 'batch_size': 4},
#     # 'MODELS_FINAL_EVAL/best_trained_model_5.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4}
    # }

# Mapping of model paths to architectures
model_info_dict = {
    'MODELS_FINAL_DEPLOY/best_trained_model_1.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
    'MODELS_FINAL_DEPLOY/best_trained_model_2.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
    'MODELS_FINAL_DEPLOY/best_trained_model_3.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
    'MODELS_FINAL_DEPLOY/best_trained_model_4.pth': {'architecture': 'EfficientNetB0', 'img_size': 1000, 'batch_size': 4},
    'MODELS_FINAL_DEPLOY/best_trained_model_5.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 8}
    }

def extract_true_labels(models_dict, device):
    """
    Extract true labels from the test dataset using the first model's test loader.
    Assumes that all test loaders iterate over the dataset in the same order.
    """
    first_model_path, first_config = next(iter(models_dict.items()))
    img_size = first_config['img_size']
    batch_size = first_config['batch_size']
    
    # Initialize test loader
    _, test_loader, _ = get_dataloaders_final_train(batch_size, img_size)
    
    all_labels = []
    with torch.no_grad():
        for _, labels in tqdm(test_loader, desc='Collecting True Labels', leave=False):
            all_labels.append(labels.numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    return all_labels

def ensemble_predict(models_dict, device):
    
    model_predictions = []
    for idx, (model_path, config) in enumerate(models_dict.items(), 1):
        model_name = config['architecture']
        img_size = config['img_size']
        # batch_size = config['batch_size']
        batch_size = 1
        
        _, test_loader, _ = get_dataloaders_final_train(batch_size, img_size)
        
        model = get_model(model_name, num_classes=1)  
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  
        
        preds = []
        with torch.no_grad():
            #progress_bar = tqdm(test_loader, desc=f'Predicting with {model}', leave=False)
            #for inputs, _ in progress_bar:
            inputs = test_loader.dataset[0].to(device, non_blocking=True)

            # Mixed precision inference
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(inputs)  # Shape: (batch_size, 1)

            # Apply sigmoid to get probabilities for class 1
            probs = torch.sigmoid(outputs).squeeze(1)  # Shape: (batch_size,)

            preds.append(probs.cpu())

        # Concatenate all batch predictions for the current model
        preds = torch.cat(preds, dim=0)  
        # model_predictions.append(preds)
        
        # Convert probabilities to predictions
        HIGH_CONFIDENCE_THRESHOLD = 0.70  
        MODERATE_CONFIDENCE_LOWER = 0.50  

        mask_high_confidence = preds >= HIGH_CONFIDENCE_THRESHOLD
        mask_moderate_confidence = (preds >= MODERATE_CONFIDENCE_LOWER) & (preds < HIGH_CONFIDENCE_THRESHOLD)
        mask_low_confidence = preds < MODERATE_CONFIDENCE_LOWER

        predicted_labels = torch.zeros_like(preds, dtype=torch.int)

        predicted_labels[mask_high_confidence] = 1
        predicted_labels[mask_moderate_confidence] = 0
        predicted_labels[mask_low_confidence] = 0

        predicted_labels_np = predicted_labels.cpu().numpy()
        model_predictions.append(list(predicted_labels_np))

    model_predictions = np.stack(model_predictions, axis=0)
    
    # Voting classifier (majority voting)
    sum_preds = np.sum(model_predictions, axis=0)  # Shape: (num_samples,)
    ensemble_preds = (sum_preds > (len(models_dict.items()) / 2)).astype(int)  # Shape: (num_samples,)

    return ensemble_preds

# Get ensemble probabilities and true labels
ensemble_preds = ensemble_predict(model_info_dict, device)
true_labels = extract_true_labels(model_info_dict, device)

# # Compute performance metrics
accuracy = accuracy_score(true_labels, ensemble_preds)
precision = precision_score(true_labels, ensemble_preds, average='macro')
recall = recall_score(true_labels, ensemble_preds, average='macro')
f1 = f1_score(true_labels, ensemble_preds, average='macro')
custom_score = calculate_custom_score(true_labels, ensemble_preds)

# Print the computed metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1 Score (Macro): {f1:.4f}")
print(f"Custom Score: {custom_score:.4f}")