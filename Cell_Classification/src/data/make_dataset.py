import os
import numpy as np
from PIL import Image
import base64
import io

def save_image_as_tif(image, filename) -> None:
    # Print the type and some information about the input image
    print(f"Input type: {type(image)}")

    # Handle different types of inputs
    if isinstance(image, str):
        print(f"String length: {len(image)}")  # Print the length of the string for more info
        if os.path.isfile(image):
            # If the input is a valid file path, open and load it as an image
            print("Detected a file path. Loading the image from the file path.")
            image = Image.open(image)
        else:
            # Check if the string might be base64 encoded image data
            try:
                # Attempt to decode the string as base64 and load it as an image
                image_data = base64.b64decode(image)
                image = Image.open(io.BytesIO(image_data))
                print("Successfully decoded base64 image data.")
            except Exception as e:
                print(f"Failed to decode base64 image data: {e}")
                raise ValueError("The provided image input appears to be a string but not a valid file path or base64 encoded image data.")
    elif isinstance(image, np.ndarray):
        # If it's a NumPy array, convert it to a PIL Image object
        print(f"Converting NumPy array of shape {image.shape} to PIL Image.")
        image = Image.fromarray(image.astype(np.uint16), mode='I;16')
    elif not isinstance(image, Image.Image):
        raise ValueError("The image should be a valid file path (string), a base64-encoded string, a NumPy array, or a PIL Image object.")

    # Define the target directory
    target_directory = 'data/validation16bit'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        
    # Construct the complete file path
    file_path = os.path.join(target_directory, f"{filename}")
    
    # Save the image as a 16-bit grayscale TIFF
    if isinstance(image, Image.Image):
        image.save(file_path, format='TIFF')
        print(f"Image saved at: {file_path}")
    else:
        raise TypeError("The final image object is not a PIL Image and cannot be saved.")

# Example usage:
# Assuming the image is a NumPy array of 16-bit grayscale values
# np_image = np.random.randint(0, 65535, (256, 256), dtype=np.uint16)
# save_image_as_tif(np_image, "example_image")
