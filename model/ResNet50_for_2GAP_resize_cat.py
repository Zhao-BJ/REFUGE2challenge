import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from skimage import io
from utils.trainer_utils import norm_image
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


pretrain_root = '/home/ubuntu/zhaobenjian/pretrained_model/'


# Pretrained model
premodel_dir = os.path.join(pretrain_root, 'pytorch')
resnet50 = os.path.join(pretrain_root, 'pytorch/resnet50-19c8e357.pth')
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, pretrained=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_conv3 = nn.Conv2d(1024, 256, 1, 1, 0, bias=False)
        self.out_conv4 = nn.Conv2d(2048, 512, 1, 1, 0, bias=False)
        self.fc_last = nn.Linear(768, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _load_pretrained_model(self):
        pretrain_dict = load_state_dict_from_url(model_urls["resnet50"], model_dir=premodel_dir, progress=False)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        feat3_conv = self.out_conv3(feat3)
        feat4_conv = self.out_conv4(feat4)
        feat4_conv = F.interpolate(feat4_conv, size=feat3.size()[2:], mode='bilinear', align_corners=True)
        feat_conv = torch.cat([feat3_conv, feat4_conv], dim=1)
        feat_fc = self.avgpool(feat_conv)
        feat_fc = feat_fc.reshape(feat_fc.size(0), -1)
        feat_fc = self.fc_last(feat_fc)
        return feat_fc, feat_conv


def _resnet(block, layers, num_classes, pretrained, **kwargs):
    model = ResNet(block, layers, num_classes, pretrained, **kwargs)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    model = _resnet(Bottleneck, [3, 4, 6, 3], num_classes, pretrained, **kwargs)
    return model


def get_cam(module, pred, feat):
    weight = list(module.weight)
    weight = weight[pred]
    feat = torch.squeeze(feat)
    cam = feat * weight[:, np.newaxis, np.newaxis]
    cam = torch.sum(cam, dim=0)
    zero_tensor = torch.zeros(size=cam.size()).cuda()
    cam = torch.max(cam, other=zero_tensor)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)
    return cam


def get_cam_threshold_save(module, pred, feat, threshold=0, draw=False, img=None, name=None, save_dir=None, mark=None):
    weight = list(module.weight)
    weight = weight[pred]
    feat = torch.squeeze(feat)
    cam = feat * weight[:, np.newaxis, np.newaxis]
    cam = torch.sum(cam, dim=0)
    zero_tensor = torch.zeros(size=cam.size()).cuda()
    cam = torch.max(cam, other=zero_tensor)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)
    cam[cam <= threshold] = 0
    if draw:
        img_size = img.size(2)
        img = img.cpu().numpy()
        img = np.float32(img)

        cam_img = cam.cpu().data.numpy()
        cam_img = np.uint8(255 * cam_img)
        cam_img = cv2.resize(cam_img, (img_size, img_size))
        #io.imsave(save_dir + "{}-{}-{}.png".format(name[0][:-4], mark, "feat"), cam_img)

        heatmap = cv2.applyColorMap(np.uint8(cam_img), cv2.COLORMAP_JET)
        heatmap = heatmap[..., ::-1]
        heatmap = np.float32(heatmap) / 255

        img = np.squeeze(img)
        img = np.transpose(img, (1, 2, 0))
        imgheatmap = heatmap * 0.3 + img * 0.5
        imgheatmap =norm_image(imgheatmap)
        io.imsave(save_dir + "{}-{}.png".format(name[:-4], mark), imgheatmap)
        heatmap = np.uint8(heatmap * 255)
        io.imsave(save_dir + "{}-{}-{}.png".format(name[:-4], mark, "heatmap"), heatmap)
    return cam
