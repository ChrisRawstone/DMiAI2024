import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Global Average Pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch_size, channels)

        # Channel Attention
        avg_out = self.fc2(F.relu(self.fc1(avg_pool))).view(batch_size, channels, 1, 1)
        max_out = self.fc2(F.relu(self.fc1(max_pool))).view(batch_size, channels, 1, 1)
        attention = torch.sigmoid(avg_out + max_out)

        return x * attention

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet, self).__init__()

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Extended bottleneck with an additional layer
        self.bottleneck = nn.Sequential(
            self.conv_block(512, 1024),
            self.conv_block(1024, 1024)  # Added an extra layer in the bottleneck
        )

        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)

        # Adding an extra layer to each decoder block
        self.decoder4 = nn.Sequential(
            self.decoder4,
            self.conv_block(512, 512)
        )
        self.decoder3 = nn.Sequential(
            self.decoder3,
            self.conv_block(256, 256)
        )
        self.decoder2 = nn.Sequential(
            self.decoder2,
            self.conv_block(128, 128)
        )
        self.decoder1 = nn.Sequential(
            self.decoder1,
            self.conv_block(64, 64)
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Attention layers
        self.attention1 = ChannelAttention(64)
        self.attention2 = ChannelAttention(128)
        self.attention3 = ChannelAttention(256)
        self.attention4 = ChannelAttention(512)

        # Additional convolution layers after concatenation
        self.conv_after_concat4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv_after_concat3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_after_concat2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_after_concat1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)  # Dropout layer added for regularization
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc1 = self.attention1(enc1)  # Attention after encoder block 1
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc2 = self.attention2(enc2)  # Attention after encoder block 2
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc3 = self.attention3(enc3)  # Attention after encoder block 3
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))
        enc4 = self.attention4(enc4)  # Attention after encoder block 4

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        # Decoder path with residual connections
        dec4 = self.decoder4(bottleneck)
        dec4 = self.concat_with_skip(enc4, dec4)  # Skip connection
        dec4 = self.conv_after_concat4(dec4)  # Convolution after concatenation

        dec3 = self.decoder3(dec4)
        dec3 = self.concat_with_skip(enc3, dec3)  # Skip connection
        dec3 = self.conv_after_concat3(dec3)  # Convolution after concatenation

        dec2 = self.decoder2(dec3)
        dec2 = self.concat_with_skip(enc2, dec2)  # Skip connection
        dec2 = self.conv_after_concat2(dec2)  # Convolution after concatenation

        dec1 = self.decoder1(dec2)
        dec1 = self.concat_with_skip(enc1, dec1)  # Skip connection
        dec1 = self.conv_after_concat1(dec1)  # Convolution after concatenation

        return self.final_conv(dec1)

    def concat_with_skip(self, enc, dec):
        # Ensure that the dimensions match for concatenation
        if enc.shape[2:] != dec.shape[2:]:
            dec = F.interpolate(dec, size=enc.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat((dec, enc), dim=1)

# Example usage
if __name__ == "__main__":
    model = UNet()
    print(model)

    # Dummy input tensor with 4 channels and size 256x256
    input_tensor = torch.randn(1, 4, 256, 256)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)

