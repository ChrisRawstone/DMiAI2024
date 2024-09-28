import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# model = models.densenet121(pretrained=True)
# model = models.efficientnet_b0(pretrained=True)
# model = models.inception_v3(pretrained=True)
# model = models.vgg16(pretrained=True)
# model = models.mobilenet_v2(pretrained=True)



class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        # model = models.inception_v3(weights='IMAGENET1K_V1')
        model = models.resnet18(weights=True)
        # Unfreeze layers starting from 'layer4'
        for name, param in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Replace the fully connected layer (classifier)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # 2 classes in your binary classification task

        # Set the device to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        
    def forward(self, x):
        return self.model(x)


# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, latent_dim)  # Adjust this based on image size
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 28 * 28),
            nn.Unflatten(1, (256, 28, 28)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output pixel values in the range [0, 1]
        )

    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed = self.decoder(latent_space)
        return reconstructed, latent_space

class ClassifierOnAE(nn.Module):
    def __init__(self, latent_dim=256, num_classes=2):
        super(ClassifierOnAE, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)