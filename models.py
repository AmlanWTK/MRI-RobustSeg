import torch
import torch.nn as nn
import torch.nn.functional as F

"""
IEEE Publication Codebase: Model Architecture Definitions
Target Conference: NETCRYPT 2026

Description:
    Defines the segmentation architectures used in the ablation study.
    Primary model: U-Net++ with nested skip connections.
    Reference: Zhou et al., "UNet++: A Nested U-Net Architecture for
    Medical Image Segmentation," MICCAI 2018.
"""

class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: (Conv2d -> BatchNorm -> ReLU) * 2
    This is the building block for the encoder and decoder paths.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    """
    U-Net++ Architecture
    Implements nested skip connections to reduce the semantic gap between 
    encoder and decoder feature maps.
    Reference: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
    """
    def __init__(self, in_ch=4, out_ch=3):  
        super().__init__()
        filters = [32, 64, 128, 256, 512]

        # Encoder
        self.conv0_0 = ConvBlock(in_ch, filters[0])
        self.pool0 = nn.MaxPool2d(2)
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4_0 = ConvBlock(filters[3], filters[4])

        # Decoder with nested skip connections
        self.up3_1 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.conv3_1 = ConvBlock(filters[3]*2, filters[3])

        self.up2_2 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.conv2_2 = ConvBlock(filters[2]*2, filters[2])

        self.up1_3 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.conv1_3 = ConvBlock(filters[1]*2, filters[1])

        self.up0_4 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.conv0_4 = ConvBlock(filters[0]*2, filters[0])

        # Output
        self.out = nn.Conv2d(filters[0], out_ch, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        x4_0 = self.conv4_0(self.pool3(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3_1(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up2_2(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up1_3(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up0_4(x1_3)], dim=1))

        return self.out(x0_4)
