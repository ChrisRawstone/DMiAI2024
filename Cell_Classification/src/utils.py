# utils.py

import numpy as np
import cv2
import base64
from PIL import Image  # For saving images in specific formats
import os

def decode_image(encoded_img: str) -> np.ndarray:
    """
    Decodes a base64 encoded image string to a NumPy array.

    Args:
        encoded_img (str): Base64 encoded image string.

    Returns:
        np.ndarray: Decoded image as a NumPy array.
    """
    try:
        # Decode the base64 string to bytes
        img_data = base64.b64decode(encoded_img)
        # Convert bytes data to a NumPy array
        np_arr = np.frombuffer(img_data, np.uint8)
        # Decode the image data using OpenCV
        image = cv2.imdecode(np_arr, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image decoding resulted in None.")
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")



    # Convert NumPy array to PIL Image with mode 'I;16'
    pil_image = Image.fromarray(image_gray.astype(np.uint16), mode='I;16')

    # Save as .tif in 'I;16B' format
    pil_image.save(output_path, format='TIFF')

def load_sample(encoded_img: str) -> dict:
    """
    Loads and decodes the sample image.

    Args:
        encoded_img (str): Base64 encoded image string.

    Returns:
        dict: Dictionary containing the image.
    """
    image = decode_image(encoded_img)  # Decode the image
    return {
        "image": image
    }

# Example usage for saving images (not part of the utils module)
if __name__ == "__main__":
    encoded_image_str = "<YOUR BASE64 ENCODED STRING HERE>"
    output_file_path = "output_image.tif"

    # Load and decode the sample
    sample = load_sample(encoded_image_str)
    image = sample["image"]

    # Save the decoded image as a .tif file
    save_image_as_tif(image, output_file_path)
    print(f"Image saved at {output_file_path}")
