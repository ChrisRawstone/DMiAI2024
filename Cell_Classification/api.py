# api.py

import uvicorn
from fastapi import FastAPI, HTTPException
import datetime
import time

from loguru import logger
from pydantic import BaseModel
from typing import List
from src.utils import load_sample
from src.data.make_dataset import save_image_as_tif
import numpy as np
import os
import random

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
        # Decode and load the image
        encoded_image = request.cell
        # image = sample.get("image")
        
        # if image is None:
        #     raise ValueError("Image decoding failed.")

        # # Ensure the image is in the correct format 
        # if not isinstance(image, np.ndarray):
        #     raise TypeError("Decoded image is not a NumPy array.")
        
        counter += 1
        # Save the image as a 16-bit .tif file
        save_path = os.path.join("", f"{counter}".zfill(3)+".tif")
        save_image_as_tif(encoded_image, save_path)


        # Make prediction by passing the image array
        # predicted_homogenous_state = predict(image)
        is_homogenous = random.randint(0, 1)

        # predicted_homogenous_state = 1
        
        # Return the prediction
        response = CellClassificationPredictResponseDto(
            is_homogenous=is_homogenous
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
