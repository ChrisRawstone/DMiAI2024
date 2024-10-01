import torch.nn as nn
import torch
import torchvision.models

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, use_attention=False):
        super(UNet, self).__init__()
        self.use_attention = use_attention

        # Encoder
        self.enc_conv0 = self.conv_block(in_channels, 64)
        self.pool0 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv1 = self.conv_block(64, 128)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv2 = self.conv_block(128, 256)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv3 = self.conv_block(256, 512)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv4 = self.conv_block(512, 1024)

        # Self-Attention layer for the bottleneck
        if self.use_attention:
            self.self_attention = SelfAttention(1024)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv3 = self.conv_block(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(512, 256)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv1 = self.conv_block(256, 128)

        self.upconv0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv0 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels,kernel_size=3):
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),  
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
        return block

    def forward(self, x):
        # Encoder
        enc0 = self.enc_conv0(x)
        enc0_pool = self.pool0(enc0)

        enc1 = self.enc_conv1(enc0_pool)
        enc1_pool = self.pool1(enc1)

        enc2 = self.enc_conv2(enc1_pool)
        enc2_pool = self.pool2(enc2)

        enc3 = self.enc_conv3(enc2_pool)
        enc3_pool = self.pool3(enc3)

        bottleneck = self.enc_conv4(enc3_pool)

        # Apply self-attention in the bottleneck if specified
        if self.use_attention:
            bottleneck = self.self_attention(bottleneck)

        # Decoder
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec_conv3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec_conv2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec_conv1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, enc0), dim=1)
        dec0 = self.dec_conv0(dec0)

        out = self.final_conv(dec0)
        return out

class UNet_ver_2(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, use_attention=False):
        super(UNet_ver_2, self).__init__()
        self.use_attention = use_attention

        # Encoder
        self.enc_conv0 = self.conv_block(in_channels, 64, kernel_size=9)
        self.pool0 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv1 = self.conv_block(64, 128, kernel_size=7)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv2 = self.conv_block(128, 256 , kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv3 = self.conv_block(256, 512, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv4 = self.conv_block(512, 1024)

        # Self-Attention layer for the bottleneck
        if self.use_attention:
            self.self_attention = SelfAttention(1024)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv3 = self.conv_block(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(512, 256, kernel_size=3)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv1 = self.conv_block(256, 128, kernel_size=5)

        self.upconv0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv0 = self.conv_block(128, 64, kernel_size=7)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels,kernel_size=3):
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),  
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
        return block

    def forward(self, x):
        # Encoder
        enc0 = self.enc_conv0(x)
        enc0_pool = self.pool0(enc0)

        enc1 = self.enc_conv1(enc0_pool)
        enc1_pool = self.pool1(enc1)

        enc2 = self.enc_conv2(enc1_pool)
        enc2_pool = self.pool2(enc2)

        enc3 = self.enc_conv3(enc2_pool)
        enc3_pool = self.pool3(enc3)

        bottleneck = self.enc_conv4(enc3_pool)

        # Apply self-attention in the bottleneck if specified
        if self.use_attention:
            bottleneck = self.self_attention(bottleneck)

        # Decoder
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec_conv3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec_conv2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec_conv1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, enc0), dim=1)
        dec0 = self.dec_conv0(dec0)

        out = self.final_conv(dec0)
        return out



    
# Load pre-trained VGG-19 model for perceptual loss
class VGG19Features(nn.Module):
    def __init__(self, layers):
        super(VGG19Features, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True).features
        self.layers = layers
        self.vgg_layers = nn.Sequential(*list(vgg19[:max(layers)+1]))
        for param in self.vgg_layers.parameters():
            param.requires_grad = False  # Freeze VGG weights

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

# Define the perceptual loss function
class PerceptualLoss(nn.Module):
    def __init__(self, layers=[2, 7, 12, 21, 30]): # Some  defualt chosen layers
        # available CONV layers_idx
        available_conv_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        assert all([layer in available_conv_indices for layer in layers]), "Invalid layer index, it is not a convolutional layer"
        
        super(PerceptualLoss, self).__init__()
        self.vgg_features = VGG19Features(layers).eval()

    def forward(self, pred, target):
        pred_features = self.vgg_features(pred)
        target_features = self.vgg_features(target)
        loss = 0.0
        for pf, tf in zip(pred_features, target_features):
            loss += torch.nn.functional.mse_loss(pf, tf)  # Perceptual loss (MSE between features)
        return loss