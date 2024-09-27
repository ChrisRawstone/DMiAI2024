import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleClassifier, self).__init__()
        # Use a pre-trained model like ResNet18 for image classification
        self.model = models.resnet18(weights=True)
        # Modify the final layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)