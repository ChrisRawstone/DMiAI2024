import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights
import torch.optim as optim
import optuna
import logging
import tqdm
import numpy as np
import json

from src.utils import calculate_custom_score
from data.make_dataset import get_dataloaders

def get_models(num_classes=1):
    """
    Returns a dictionary of state-of-the-art models.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 1.

    Returns:
        dict: Dictionary of models.
    """
    models_dict = {}

    # Vision Transformer
    vit_weights = ViT_B_16_Weights.DEFAULT
    vit_model = models.vit_b_16(weights=vit_weights)
    in_features = vit_model.heads.head.in_features
    vit_model.heads.head = nn.Linear(in_features, num_classes)
    models_dict['ViT'] = vit_model

    # ResNet50
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(in_features, num_classes)
    models_dict['ResNet50'] = resnet_model

    # EfficientNet
    efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = efficientnet_model.classifier[1].in_features
    efficientnet_model.classifier[1] = nn.Linear(in_features, num_classes)
    models_dict['EfficientNet'] = efficientnet_model

    # MobileNetV3
    mobilenet_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = mobilenet_model.classifier[3].in_features
    mobilenet_model.classifier[3] = nn.Linear(in_features, num_classes)
    models_dict['MobileNetV3'] = mobilenet_model

    return models_dict

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True):
        """
        Initializes the Focal Loss function.

        Args:
            alpha (float, optional): Weighting factor for the rare class. Defaults to 0.25.
            gamma (float, optional): Focusing parameter for modulating factor (1-p). Defaults to 2.
            logits (bool, optional): Whether inputs are logits. Defaults to True.
            reduce (bool, optional): Whether to reduce the loss. Defaults to True.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss.

        Args:
            inputs (torch.Tensor): Predicted logits.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        targets = targets.type_as(inputs)
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def get_model_parallel(model):
    """
    Wraps the model with DataParallel if multiple GPUs are available.

    Args:
        model (nn.Module): The model to wrap.

    Returns:
        nn.Module: The potentially parallelized model.
    """
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    else:
        logging.info("Using a single GPU.")
    return model

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Validation custom score to maximize.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters to tune
    model_name = trial.suggest_categorical('model_name', ['ViT', 'ResNet50', 'EfficientNet', 'MobileNetV3'])

    if model_name == 'ViT':
        img_size = 224  # ViT only supports 224x224 images
    else:
        img_size = trial.suggest_categorical('img_size', [224, 256, 299])

    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 1.0, 3.0)
    alpha = trial.suggest_float('alpha', 0.1, 0.9)

    num_epochs = 50 # Fixed number of epochs
    
    train_loader, val_loader = get_dataloaders(batch_size, img_size)

    # Get model
    models_dict = get_models()
    model = models_dict[model_name]
    model = get_model_parallel(model)
    model = model.to(device)

    # Loss function
    criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler("cuda")  # Updated to use torch.cuda.amp

    best_custom_score = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):  # Updated autocast usage
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.detach().cpu().numpy())

            progress_bar.set_postfix({'Loss': loss.item()})

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation', leave=False)
            for images, labels in progress_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):  # Updated autocast usage
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs).detach().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.detach().cpu().numpy())

        # Calculate custom score
        preds_binary = (np.array(val_preds) > 0.5).astype(int)
        custom_score = calculate_custom_score(val_targets, preds_binary)

        scheduler.step()

        # Update best score and save model + JSON
        if custom_score > best_custom_score:
            best_custom_score = custom_score

            # Save the best model from Optuna hypertuning
            save_path_optuna = 'checkpoints/best_model_optuna.pth'
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path_optuna)
            else:
                torch.save(model.state_dict(), save_path_optuna)
            logging.info(f"New best Optuna model saved with Val Score: {best_custom_score:.4f}")

            # Save model architecture and hyperparameters for Optuna best model
            model_info_optuna = {
                'model_name': model_name,
                'img_size': img_size,
                'batch_size': batch_size,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'gamma': gamma,
                'alpha': alpha
            }
            with open('checkpoints/model_info_optuna.json', 'w') as f:
                json.dump(model_info_optuna, f, indent=4)
            logging.info("Optuna best model architecture and hyperparameters saved to 'checkpoints/model_info_optuna.json'.")

        # Report intermediate objective value
        trial.report(custom_score, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_custom_score