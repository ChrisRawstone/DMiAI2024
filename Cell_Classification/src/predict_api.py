import base64
import io
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from src.utils import get_model

from src.data.make_dataset import convert_16bit_to_8bit  

def get_image(image) -> np.ndarray:
    
    image_data = base64.b64decode(image)
    image = Image.open(io.BytesIO(image_data))
    image_array = np.array(image, dtype=np.uint16)
    
    return image_array

def transform_image(image, img_size):

    image = get_image(image)
    image = convert_16bit_to_8bit(image)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.485, 0.485),  # Using ImageNet means
                    std=(0.229, 0.229, 0.229)),   # Using ImageNet stds
        ToTensorV2()])

    transformed = val_transform(image=image)
    image = transformed['image']
    return image 

def ensemble_predict(models_dict, image, device):
    
    model_predictions = []
    for idx, (model_path, config) in enumerate(models_dict.items(), 1):
        model_name = config['architecture']
        img_size = config['img_size']
        
        image_tensor = transform_image(image, img_size)
                
        model = get_model(model_name, num_classes=1)  
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  
                
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                # # Add a batch dimension by unsqueezing at dim=0
                # inputs = image_tensor.to(device, non_blocking=True)

                # # Enable mixed precision inference if desired
                # outputs = model(inputs.unsqueeze(0)) 

                # # Apply sigmoid to get the probability for class 1
                # probs = torch.sigmoid(outputs).detach().cpu().numpy()

                # # Move the probability to CPU and convert to a Python scalar
                # pred_prob = probs# .cpu().item()
                inputs = image_tensor.to(device, non_blocking=True)
                outputs = model(inputs.unsqueeze(0))
                probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
                pred_prob = probs.item()  # If you expect a scalar prediction
        
        # Convert probabilities to predictions
        HIGH_CONFIDENCE_THRESHOLD = 0.70  
        MODERATE_CONFIDENCE_LOWER = 0.50  

        mask_high_confidence = pred_prob >= HIGH_CONFIDENCE_THRESHOLD
        mask_moderate_confidence = (pred_prob >= MODERATE_CONFIDENCE_LOWER) & (pred_prob < HIGH_CONFIDENCE_THRESHOLD)
        mask_low_confidence = pred_prob < MODERATE_CONFIDENCE_LOWER

        predicted_labels = torch.zeros_like(pred_prob, dtype=torch.int)

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

# model_info_dict = {
#     'MODELS_FINAL_DEPLOY/best_trained_model_1.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
#     'MODELS_FINAL_DEPLOY/best_trained_model_2.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
#     'MODELS_FINAL_DEPLOY/best_trained_model_3.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
#     'MODELS_FINAL_DEPLOY/best_trained_model_4.pth': {'architecture': 'EfficientNetB0', 'img_size': 1000, 'batch_size': 4},
#     'MODELS_FINAL_DEPLOY/best_trained_model_5.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 8}
#     }

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ensemble_preds = ensemble_predict(model_info_dict, image, device)