# models/model.py

import torch
import torch.nn as nn
from torchvision import models

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        # Pretrained ResNet18 Encoder
        pretrained_encoder = models.resnet18(pretrained=True)
        # Remove the final fully connected layer and average pooling
        self.encoder = nn.Sequential(*list(pretrained_encoder.children())[:-1])  # Output: (batch_size, 512, 1, 1)
        
        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.fc_enc = nn.Linear(pretrained_encoder.fc.in_features, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 512 * 1 * 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 1x1 -> 2x2
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 2x2 -> 4x4
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 4x4 -> 8x8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 8x8 -> 16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 16x16 -> 32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),     # 32x32 -> 64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),      # 64x64 -> 128x128
            nn.Sigmoid(),  # Ensures output is between 0 and 1
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)             # (batch_size, 512, 1, 1)
        encoded = torch.flatten(encoded, 1)    # (batch_size, 512)
        latent = self.fc_enc(encoded)          # (batch_size, latent_dim)
        
        # Decode
        decoded = self.fc_dec(latent)          # (batch_size, 512)
        decoded = decoded.view(-1, 512, 1, 1)  # (batch_size, 512, 1, 1)
        reconstructed = self.decoder(decoded)  # (batch_size, 3, 128, 128)
        return reconstructed


class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=2):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        return self.classifier(x)
