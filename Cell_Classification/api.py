# api.py

import uvicorn
from fastapi import FastAPI, HTTPException
import datetime
import time
from src.utils import load_model, save_image_as_tif  # Updated import
from src.predict import predict
from loguru import logger
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import joblib
from src.data.make_dataset import tif_to_ndarray, convert_16bit_to_8bit  # Updated import
import matplotlib.pyplot as plt

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


# Load the trained model
model_path = "models/svm_model_optimized.pkl"  # Path to the saved model
model = joblib.load(model_path)
print(f"Model loaded from {model_path}")

@app.post('/predict', response_model=CellClassificationPredictResponseDto)
def predict_endpoint(request: CellClassificationPredictRequestDto):
    global counter
    try:
        # Decode and load the image
        image = tif_to_ndarray(request.cell)

        image = convert_16bit_to_8bit(image)

        plt.imshow(image)
        plt.axis('off')
        plt.savefig(f'plots/test.png')
        plt.show()

        
        predicted_homogenous_state=0
        
        # Return the prediction
        response = CellClassificationPredictResponseDto(
            is_homogenous=predicted_homogenous_state
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