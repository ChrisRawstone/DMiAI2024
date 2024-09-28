# utils.py

import numpy as np
import cv2
import base64 

def decode_image(encoded_img: str) -> np.ndarray:
    """
    Decodes a base64 encoded image string to a NumPy array.

    Args:
        encoded_img (str): Base64 encoded image string.

    Returns:
        np.ndarray: Decoded image.
    """
    try:
        # Decode the base64 string to bytes
        img_data = base64.b64decode(encoded_img)
        # Convert bytes data to NumPy array
        np_arr = np.frombuffer(img_data, np.uint8)
        # Decode the image data using OpenCV
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image decoding resulted in None.")
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

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
