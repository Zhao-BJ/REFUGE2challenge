import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    def __init__(self, inplanes, planes):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class downsample(nn.Module):
    def __init__(self, inplanes, planes):
        super(downsample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            double_conv(inplanes=inplanes, planes=planes)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class upsample(nn.Module):
    def __init__(self, inplanes, planes):
        super(upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv(inplanes=inplanes, planes=planes)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.in_conv = double_conv(inplanes=3, planes=64)
        self.downsample1 = downsample(inplanes=64, planes=128)
        self.downsample2 = downsample(inplanes=128, planes=256)
        self.downsample3 = downsample(inplanes=256, planes=512)
        self.downsample4 = downsample(inplanes=512, planes=512)
        self.upsample1 = upsample(inplanes=1024, planes=256)
        self.upsample2 = upsample(inplanes=512, planes=128)
        self.upsample3 = upsample(inplanes=256, planes=64)
        self.upsample4 = upsample(inplanes=128, planes=64)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        x4 = self.downsample3(x3)
        x5 = self.downsample4(x4)
        x = self.upsample1(x5, x4)
        x = self.upsample2(x, x3)
        x = self.upsample3(x, x2)
        x = self.upsample4(x, x1)
        x = self.out_conv(x)
        # x = torch.sigmoid(x)
        return x
