import base64
import io
import numpy as np
import json
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
    image = image.astype(np.float32) / 255.0
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.485, 0.485),  # Using ImageNet means
                    std=(0.229, 0.229, 0.229)),   # Using ImageNet stds
        ToTensorV2()])

    transformed = val_transform(image=image)

    image = transformed['image']

    return image 

def load_model(checkpoint_path, model_info_path, device):
    """
    Loads the model architecture and weights.

    Args:
        checkpoint_path (str): Path to the model weights.
        model_info_path (str): Path to the model architecture info JSON.
        device (torch.device): Device to load the model on.

    Returns:
        tuple: (model, img_size, model_info)
    """
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_name = model_info['model_name']
    img_size = model_info['img_size']

    model = get_model(model_name, num_classes=1)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, img_size, model_info

def ensemble_predict(models_dict, image_type,  device):
    model_predictions = []
    preds = []
    for idx, (model_path, config) in enumerate(models_dict.items(), 1):
        model_name = config['architecture']
        img_size = config['img_size']

        image = transform_image(image_type, img_size)
        
        model = get_model(model_name, num_classes=1)  
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  
        
        inputs = image.to(device, non_blocking=True)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(inputs.unsqueeze(0))  

            # Apply sigmoid to get probabilities for class 1
            probs = torch.sigmoid(outputs).squeeze(1)  

            preds.append(probs.cpu())

    preds = torch.cat(preds, dim=0)  
        
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
    sum_preds = np.sum(model_predictions[0], axis=0)  # Shape: (num_samples,)
    ensemble_preds = (sum_preds > (len(models_dict.items()) / 2)).astype(int)  # Shape: (num_samples,)

    return ensemble_preds

# if __name__ == '__main__':
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
#     model_checkpoint = 'MODELS_FINAL_EVAL/best_trained_model_2.pth'
#     model_info = 'MODELS_FINAL_EVAL/best_model_2.json'

#     model, img_size, model_info = load_model(model_checkpoint, model_info, device)

#     model_info_dict = {
#         'MODELS_FINAL_DEPLOY/best_trained_model_1.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
#         'MODELS_FINAL_DEPLOY/best_trained_model_2.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
#         'MODELS_FINAL_DEPLOY/best_trained_model_3.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
#         'MODELS_FINAL_DEPLOY/best_trained_model_4.pth': {'architecture': 'EfficientNetB0', 'img_size': 1000, 'batch_size': 4},
#         'MODELS_FINAL_DEPLOY/best_trained_model_5.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 8}
#         }
    
#     preds_binary = ensemble_predict_2(model_info_dict, device='cuda')

#     print(preds_binary)