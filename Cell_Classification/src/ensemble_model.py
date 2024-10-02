import copy
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from collections import Counter
from data.make_dataset import get_dataloaders_final_train, LoadTifDataset, get_dataloaders
from src.utils import calculate_custom_score
from models.model import FocalLoss, LabelSmoothingLoss, BalancedBCELoss, get_model_parallel
from utils import get_model

model_path_to_architecture = {
    'models/best_model_1.pth': 'resnet18',
    'models/best_model_2.pth': 'vgg16',
    'models/best_model_3.pth': 'efficientnet_b0',
    'models/best_model_4.pth': 'resnet18',
    'models/best_model_5.pth': 'vgg16'
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dictionary to store models
models = {}
for path, model_name in model_path_to_architecture.items():
    model = get_model(model_name)  
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()  
    # Store in the dictionary
    models[path] = model


