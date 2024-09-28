import numpy as np
import random
import torch
from torchvision import transforms
from src.models.model import SimpleClassifier
from src.data.make_dataset import PadToSize

def predict(image: np.ndarray) -> int:

    model = SimpleClassifier(num_classes=2)  
    model.load_state_dict(torch.load('/dtu/blackhole/00/156512/DM-i-AI-2024/Cell_Classification/resnet16_dummy.pth'))
    model.eval()

    target_width = 1500
    target_height = 1470
    transform = transforms.Compose([
                                    PadToSize(target_width=target_width, target_height=target_height),  # Custom padding transform
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor()])
    
    image = transform(image)
    
    is_homogenous = model(image)

    return is_homogenous


# def example_model(image) -> int:

#     is_homogenous = random.randint(0, 1)

#     return is_homogenous

def test_predict():
    image = np.random.rand(1470, 1500, 3)
    assert predict(image) in [0, 1]

test_predict()

