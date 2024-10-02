import base64
import io
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.data.make_dataset import convert_16bit_to_8bit  

def get_image(image) -> np.ndarray:
    
    image_data = base64.b64decode(image)
    image = Image.open(io.BytesIO(image_data))
    image_array = np.array(image, dtype=np.uint16)
    
    return image_array

def transform_image(image):

    image = get_image(image)
    image = convert_16bit_to_8bit(image)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img_size = 124 # remeber to take this from the json congfig
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.485, 0.485),  # Using ImageNet means
                    std=(0.229, 0.229, 0.229)),   # Using ImageNet stds
        ToTensorV2()])

    transformed = val_transform(image=image)
    image = transformed['image']

    return image 

def predict(model, image_tensor, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    image_tensor = image_tensor.to(device, non_blocking=True)  # Move image to the specified device

    with torch.no_grad():
        # Use autocast for mixed precision inference if device is cuda
        with torch.amp.autocast(device_type=device):
            outputs = model(image_tensor.unsqueeze(0))  # Add batch dimension

        # Apply sigmoid if it's a binary classification problem
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        preds_binary = (np.array(preds) > 0.5).astype(int)
        
    return preds_binary