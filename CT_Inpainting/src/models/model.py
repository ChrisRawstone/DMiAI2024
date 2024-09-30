import torch.nn as nn
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention mechanism
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# U-Net with attention
class UNetWithAttention(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNetWithAttention, self).__init__()
        self.enc_conv0 = self.conv_block(in_channels, 64)
        self.pool0 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv1 = self.conv_block(64, 128)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv2 = self.conv_block(128, 256)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv3 = self.conv_block(256, 512)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv4 = self.conv_block(512, 1024)

        # Decoder with attention gates
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attention3 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec_conv3 = self.conv_block(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attention2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec_conv2 = self.conv_block(512, 256)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attention1 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec_conv1 = self.conv_block(256, 128)

        self.upconv0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attention0 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec_conv0 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
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

        # Decoder with attention
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((self.attention3(dec3, enc3), dec3), dim=1)
        dec3 = self.dec_conv3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((self.attention2(dec2, enc2), dec2), dim=1)
        dec2 = self.dec_conv2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((self.attention1(dec1, enc1), dec1), dim=1)
        dec1 = self.dec_conv1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((self.attention0(dec0, enc0), dec0), dim=1)
        dec0 = self.dec_conv0(dec0)

        out = self.final_conv(dec0)
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = self.conv_block(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = self.conv_block(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = self.conv_block(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = self.conv_block(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, padding=1)  # Patch output
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.sigmoid(x)  # Shape: [batch_size, 1, N, N] where N is the number of patches




# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet, self).__init__()
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

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
    
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.conv1 = self.conv_block(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = self.conv_block(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = self.conv_block(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = self.conv_block(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, padding=0)  # Output a single scalar
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.sigmoid(x)
