from PIL import Image

# Define the path to the sample image
image_path = "data/training/001.tif"

# Open the image and check its mode
img = Image.open(image_path)
print(f"Original Image Mode: {img.mode}")  # Print the image mode (e.g., L, RGB, RGBA, etc.)

# Convert to RGB if needed
if img.mode != "RGB":
    img = img.convert("RGB")

# Save the image for inspection
img.save("plots/converted_image.png")
print("Converted image saved successfully.")

import numpy as np

# Convert to numpy array and inspect pixel values
img_array = np.array(img)
print(f"Image Array Shape: {img_array.shape}")
print(f"Image Array Pixel Values: {img_array}")

# print max and min pixel values
print(f"Max pixel value: {np.max(img_array)}")
print(f"Min pixel value: {np.min(img_array)}")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img=mpimg.imread(image_path)
imgplot = plt.imshow(img)
# save
plt.savefig('plots/example.png')
