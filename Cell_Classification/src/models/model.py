import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B4_Weights,
    ViT_H_14_Weights,
    MobileNet_V3_Large_Weights
)
from timm import create_model
import logging

def get_models(num_classes=1):
    """
    Returns a dictionary of state-of-the-art models.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 1.

    Returns:
        dict: Dictionary of models.
    """
    models_dict = {}

    # Vision Transformer16
    vit_weights = ViT_B_16_Weights.DEFAULT
    vit_model = models.vit_b_16(weights=vit_weights)
    in_features = vit_model.heads.head.in_features
    vit_model.heads.head = nn.Linear(in_features, num_classes)
    models_dict['ViT16'] = vit_model
    
    # Vision Transformer32
    vit_weights32 = ViT_B_32_Weights.DEFAULT
    vit_model32 = models.vit_b_32(weights=vit_weights32)
    in_features = vit_model32.heads.head.in_features
    vit_model32.heads.head = nn.Linear(in_features, num_classes)
    models_dict['ViT32'] = vit_model32
    
    # ResNet
    resnet50_weights = ResNet50_Weights.DEFAULT
    resnet50_model = models.resnet50(weights=resnet50_weights)
    in_features = resnet50_model.fc.in_features
    resnet50_model.fc = nn.Linear(in_features, num_classes)
    models_dict['ResNet50'] = resnet50_model
    
    resnet18_weights = ResNet18_Weights.DEFAULT
    resnet18_model = models.resnet18(weights=resnet18_weights)
    in_features = resnet18_model.fc.in_features
    resnet18_model.fc = nn.Linear(in_features, num_classes)
    models_dict['ResNet18'] = resnet18_model
    
    resnet101_weights = ResNet101_Weights.DEFAULT
    resnet101_model = models.resnet101(weights=resnet101_weights)
    in_features = resnet101_model.fc.in_features
    resnet101_model.fc = nn.Linear(in_features, num_classes)
    models_dict['ResNet101'] = resnet101_model
    
    # EfficientNetB0
    efficientnet_b0_weights = EfficientNet_B0_Weights.DEFAULT
    efficientnet_b0_model = models.efficientnet_b0(weights=efficientnet_b0_weights)
    in_features = efficientnet_b0_model.classifier[1].in_features
    efficientnet_b0_model.classifier[1] = nn.Linear(in_features, num_classes)
    models_dict['EfficientNetB0'] = efficientnet_b0_model

    # EfficientNetB4
    efficientnet_b4_weights = EfficientNet_B4_Weights.DEFAULT
    efficientnet_b4_model = models.efficientnet_b4(weights=efficientnet_b4_weights)
    in_features = efficientnet_b4_model.classifier[1].in_features
    efficientnet_b4_model.classifier[1] = nn.Linear(in_features, num_classes)
    models_dict['EfficientNetB4'] = efficientnet_b4_model

    # MobileNetV3 Large
    mobilenet_v3_weights = MobileNet_V3_Large_Weights.DEFAULT
    mobilenet_v3_model = models.mobilenet_v3_large(weights=mobilenet_v3_weights)
    in_features = mobilenet_v3_model.classifier[3].in_features
    mobilenet_v3_model.classifier[3] = nn.Linear(in_features, num_classes)
    models_dict['MobileNetV3'] = mobilenet_v3_model

    # DenseNet121
    densenet_weights = models.DenseNet121_Weights.DEFAULT
    densenet_model = models.densenet121(weights=densenet_weights)
    in_features = densenet_model.classifier.in_features
    densenet_model.classifier = nn.Linear(in_features, num_classes)
    models_dict['DenseNet121'] = densenet_model

    # Swin Transformer
    swin_model224 = create_model('swinv2_base_window16_256', pretrained=True) 
    swin_model224.reset_classifier(num_classes=num_classes)
    models_dict['SwinTransformer_B256'] = swin_model224
    
    # Swin Transformer
    swin_model_S224 = create_model('swin_base_patch4_window7_224', pretrained=True) 
    swin_model_S224.reset_classifier(num_classes=num_classes)
    models_dict['SwinTransformer_B224'] = swin_model_S224
    
    # Swin Transformer
    swin_model384 = create_model('swin_large_patch4_window12_384', pretrained=True) 
    swin_model384.reset_classifier(num_classes=num_classes)
    models_dict['SwinTransformer_H384'] = swin_model384
    
    # Swin Transformer
    swin_modelB384 = create_model('swinv2_large_window12to24_192to384', pretrained=True)  
    swin_modelB384.reset_classifier(num_classes=num_classes)
    models_dict['SwinTransformer_L384'] = swin_modelB384
    
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