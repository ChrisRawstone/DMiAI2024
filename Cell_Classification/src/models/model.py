import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights
from torchvision.models import (
    ResNet101_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B4_Weights,
    DenseNet121_Weights,
)
import logging

def get_models(num_classes=1):
    """
    Returns a dictionary of state-of-the-art models with frozen layers except the last layer.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 1.

    Returns:
        dict: Dictionary of models.
    """
    models_dict = {}

    # Vision Transformer16
    vit_weights = ViT_B_16_Weights.DEFAULT
    vit_model = models.vit_b_16(weights=vit_weights)
    for param in vit_model.parameters():
        param.requires_grad = False
    in_features = vit_model.heads.head.in_features
    vit_model.heads.head = nn.Linear(in_features, num_classes)
    models_dict['ViT16'] = vit_model

    # Vision Transformer32
    vit_weights32 = ViT_B_32_Weights.DEFAULT
    vit_model32 = models.vit_b_32(weights=vit_weights32)
    for param in vit_model32.parameters():
        param.requires_grad = False
    in_features = vit_model32.heads.head.in_features
    vit_model32.heads.head = nn.Linear(in_features, num_classes)
    models_dict['ViT32'] = vit_model32

    # ResNet101
    resnet101_weights = ResNet101_Weights.DEFAULT
    resnet101_model = models.resnet101(weights=resnet101_weights)
    for param in resnet101_model.parameters():
        param.requires_grad = False
    in_features = resnet101_model.fc.in_features
    resnet101_model.fc = nn.Linear(in_features, num_classes)
    models_dict['ResNet101'] = resnet101_model

    # EfficientNetB0
    efficientnet_b0_weights = EfficientNet_B0_Weights.DEFAULT
    efficientnet_b0_model = models.efficientnet_b0(weights=efficientnet_b0_weights)
    for param in efficientnet_b0_model.parameters():
        param.requires_grad = False
    in_features = efficientnet_b0_model.classifier[1].in_features
    efficientnet_b0_model.classifier[1] = nn.Linear(in_features, num_classes)
    models_dict['EfficientNetB0'] = efficientnet_b0_model

    # EfficientNetB4
    efficientnet_b4_weights = EfficientNet_B4_Weights.DEFAULT
    efficientnet_b4_model = models.efficientnet_b4(weights=efficientnet_b4_weights)
    for param in efficientnet_b4_model.parameters():
        param.requires_grad = False
    in_features = efficientnet_b4_model.classifier[1].in_features
    efficientnet_b4_model.classifier[1] = nn.Linear(in_features, num_classes)
    models_dict['EfficientNetB4'] = efficientnet_b4_model

    # DenseNet121
    densenet_weights = DenseNet121_Weights.DEFAULT
    densenet_model = models.densenet121(weights=densenet_weights)
    for param in densenet_model.parameters():
        param.requires_grad = False
    in_features = densenet_model.classifier.in_features
    densenet_model.classifier = nn.Linear(in_features, num_classes)
    models_dict['DenseNet121'] = densenet_model

    return models_dict

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

# Define loss functions

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True):
        """
        Initializes the Focal Loss function.

        Args:
            alpha (float, optional): Weighting factor for the positive class. Defaults to 0.25.
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
            inputs (torch.Tensor): Predicted logits. Shape: [batch_size, 1]
            targets (torch.Tensor): Ground truth labels. Shape: [batch_size, 1]

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        targets = targets.type_as(inputs)
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        
        # Apply alpha per class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.25):  
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        targets = targets.float()
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(inputs, targets)

class BalancedBCELoss(nn.Module):
    def __init__(self):
        super(BalancedBCELoss, self).__init__()

    def forward(self, inputs, targets):
        targets = targets.float()
        beta = targets.mean()
        pos_weight = beta
        neg_weight = 1 - beta
        inputs = torch.sigmoid(inputs)
        loss = - (pos_weight * targets * torch.log(inputs + 1e-8) + 
                    neg_weight * (1 - targets) * torch.log(1 - inputs + 1e-8))
        return loss.mean()