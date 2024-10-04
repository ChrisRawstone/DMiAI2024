from PIL import Image
import torch
from simple_lama_inpainting import SimpleLama

import numpy as np

img_path = "data/corrupted/corrupted_000_2.png"
#img_path = "data/ct/ct_000_0.png"
mask_path = "data/mask/mask_000_2.png"
tissue_path = "data/tissue/tissue_000_2.png"

# Load image and mask
image = Image.open(img_path).convert('RGB')
mask = Image.open(mask_path).convert('L')

# compute the mask only where tissue is non-zero
tissue = Image.open(tissue_path).convert('L')
tissue = np.array(tissue)
mask = np.array(mask)
mask = np.where(tissue > 0, mask, 0)
mask = Image.fromarray(mask)


# Explicitly load the model on CPU
simple_lama = SimpleLama()





result = simple_lama(image, mask)

# Show the inpainted image, the mask and the original image in a figure
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

plt.subplot(132)
plt.imshow(mask, cmap='gray')
plt.title("Mask")
plt.axis('off')

plt.subplot(133)
plt.imshow(result)
plt.title("Inpainted image")
plt.axis('off')

plt.savefig("lama.png")
plt.show()

