
# Original Model Architecture (save as 'original_unet.py')
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class OriginalUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=3):
        super(OriginalUNet, self).__init__()

        # Encoder
        self.conv0_0 = ConvBlock(in_ch, 32)
        self.conv1_0 = ConvBlock(32, 64)
        self.conv2_0 = ConvBlock(64, 128)
        self.conv3_0 = ConvBlock(128, 256)
        self.conv4_0 = ConvBlock(256, 512)

        # Decoder  
        self.up3_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3_1 = ConvBlock(256 + 256, 256)

        self.up2_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  
        self.conv2_2 = ConvBlock(128 + 128, 128)

        self.up1_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1_3 = ConvBlock(64 + 64, 64)

        self.up0_4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv0_4 = ConvBlock(32 + 32, 32)

        self.out = nn.Conv2d(32, out_ch, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.maxpool(x0_0))
        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x4_0 = self.conv4_0(self.maxpool(x3_0))

        # Decoder
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3_1(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up2_2(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up1_3(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up0_4(x1_3)], 1))

        return self.out(x0_4)
