import torch
from torch import nn
import torch.nn.functional as F

class DoubleConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, intermediate_channels=None):
        super().__init__()
        intermediate_channels = intermediate_channels or output_channels
        self.double_convolution = nn.Sequential(
            nn.Conv2d(input_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_convolution(x)

class DownSamplingLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.downsample_with_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvLayer(input_channels, output_channels)
        )

    def forward(self, x):
        return self.downsample_with_conv(x)

class UpSamplingLayer(nn.Module):
    def __init__(self, input_channels, output_channels, use_bilinear=True):
        super().__init__()

        if use_bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvLayer(input_channels, output_channels, input_channels // 2)
        else:
            self.upsample = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvLayer(input_channels, output_channels)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutputLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.final_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.final_conv(x)

class UNetModel(nn.Module):
    def __init__(self, num_classes, num_channels=14, use_bilinear=True):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.use_bilinear = use_bilinear

        self.initial_conv = DoubleConvLayer(num_channels, 64)
        self.down1 = DownSamplingLayer(64, 128)
        self.down2 = DownSamplingLayer(128, 256)
        self.down3 = DownSamplingLayer(256, 512)
        factor = 2 if use_bilinear else 1
        self.down4 = DownSamplingLayer(512, 1024 // factor)
        self.up1 = UpSamplingLayer(1024, 512 // factor, use_bilinear)
        self.up2 = UpSamplingLayer(512, 256 // factor, use_bilinear)
        self.up3 = UpSamplingLayer(256, 128 // factor, use_bilinear)
        self.up4 = UpSamplingLayer(128, 64, use_bilinear)
        self.output_conv = OutputLayer(64, num_classes)

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.output_conv(x)
        return logits
