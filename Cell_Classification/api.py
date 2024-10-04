import datetime
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
import torch
from src.predict_api import ensemble_predict

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
        image = request.cell
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_info_dict = {
        'MODELS_FINAL_DEPLOY/best_trained_model_2.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
        'MODELS_FINAL_DEPLOY/best_trained_model_1.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
        'MODELS_FINAL_DEPLOY/best_trained_model_3.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 4},
        'MODELS_FINAL_DEPLOY/best_trained_model_4.pth': {'architecture': 'EfficientNetB0', 'img_size': 1000, 'batch_size': 4},
        'MODELS_FINAL_DEPLOY/best_trained_model_5.pth': {'architecture': 'EfficientNetB0', 'img_size': 1400, 'batch_size': 8}
        }

        preds_binary = ensemble_predict(model_info_dict, image, device=device)

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