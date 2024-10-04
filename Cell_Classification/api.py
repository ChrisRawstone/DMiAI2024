import datetime
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# from src.utils import load_model
from src.predict_api import ensemble_predict, transform_image
# from src.data.make_dataset import convert_16bit_to_8bit, get_image


HOST = "0.0.0.0"
PORT = 9090

global counter
counter = 0

# Define the request and response schemas
class CellClassificationPredictRequestDto(BaseModel):
    cell: str  # Base64 encoded image string

class CellClassificationPredictResponseDto(BaseModel):
    is_homogenous: int

app = FastAPI()
start_time = time.time()

@app.get('/api')
def hello():    
    uptime = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    return {
        "service": "cell-segmentation-usecase",
        "uptime": uptime
    }

@app.get('/')
def index():
    return {"message": "Your endpoint is running!"}


@app.post('/predict', response_model=CellClassificationPredictResponseDto)
def predict_endpoint(request: CellClassificationPredictRequestDto):
    global counter
    try:
        print("Request received")
        # image = get_image(request.cell)
        # image = convert_16bit_to_8bit(image)

        # if len(image.shape) == 2:
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # else:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # img_size = 350 # remeber to take this from the json congfig
        
        # val_transform = A.Compose([
        #     A.Resize(img_size, img_size),
        #     A.Normalize(mean=(0.485, 0.485, 0.485),  # Using ImageNet means
        #                 std=(0.229, 0.229, 0.229)),   # Using ImageNet stds
        #     ToTensorV2()])

        # transformed = val_transform(image=image)
        # image = transformed['image']

        image = request.cell
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}\n")

        model_info_dict = {
        'MODELS_FINAL_DEPLOY/best_trained_model_2.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
        'MODELS_FINAL_DEPLOY/best_trained_model_2.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
        'MODELS_FINAL_DEPLOY/best_trained_model_3.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
        'MODELS_FINAL_DEPLOY/best_trained_model_4.pth': {'architecture': 'EfficientNetB0', 'img_size': 1000, 'batch_size': 4},
        'MODELS_FINAL_DEPLOY/best_trained_model_5.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 8}
        }

        preds_binary = ensemble_predict(model_info_dict, image, device='cuda')

        # Return the prediction
        response = CellClassificationPredictResponseDto(
            is_homogenous=preds_binary
        )
        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT,
        reload=True  # Enable auto-reload for development; disable in production
    )