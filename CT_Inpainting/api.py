import uvicorn
from fastapi import FastAPI
import datetime
import time
from src.utils import validate_reconstruction, encode_image, decode_request
from loguru import logger
from pydantic import BaseModel

from src.model_classes.model import UNet
import torch
from src.predict_model import predict, apply_only_to_mask

HOST = "0.0.0.0"
PORT = 9090

# Images are loaded via cv2, encoded via base64 and sent as strings
# See utils.py for details
class InpaintingPredictRequestDto(BaseModel):
    corrupted_image: str 
    tissue_image: str
    mask_image: str
    vertebrae: int

class InpaintingPredictResponseDto(BaseModel):
    reconstructed_image: str

app = FastAPI()
start_time = time.time()

@app.get('/api')
def hello():
    return {
        "service": "ct-inpainting-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }

@app.get('/')
def index():
    return "Your endpoint is running!"

@app.post('/predict', response_model=InpaintingPredictResponseDto)
def predict_endpoint(request: InpaintingPredictRequestDto):

    # Decode request
    data:dict = decode_request(request)
    corrupted_image = data["corrupted_image"]
    tissue_image = data["tissue_image"]
    mask_image = data["mask_image"]
    vertebrae = data["vertebrae"]

    logger.info(f'Recieved images: {corrupted_image.shape}')


    # Set the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize or load the trained model
    model = UNet(in_channels=4, out_channels=1)  # Create an instance of the UNet model
    model.load_state_dict(torch.load("models/ct_inpainting_unet_20240928_162225.pth", map_location=device))  # Load trained weights
    model.to(device)

    # Predict reconstruction using the model
    reconstructed_image = predict(corrupted_image, tissue_image, mask_image, vertebrae, model)
    
    reconstructed_image = apply_only_to_mask(corrupted_image, tissue_image, mask_image, reconstructed_image)


    # Validate image format
    validate_reconstruction(reconstructed_image)

    # Encode the image array to a str
    encoded_reconstruction = encode_image(reconstructed_image)

    # Return the encoded image to the validation/evalution service
    response = InpaintingPredictResponseDto(
        reconstructed_image=encoded_reconstruction
    )
    logger.info(f'Returning reconstruction: {reconstructed_image.shape}')
    return response

if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )