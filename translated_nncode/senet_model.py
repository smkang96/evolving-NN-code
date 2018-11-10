'''translation of NN code to samplable format: ResNet34'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResLayer(nn.Module):
    def __init__(self, inp, outp, l_num, stride=1):
        downsample = None
        if stride != 1 or inp != outp:
            downsample = nn.Sequential(
                conv1x1(inp, outp, stride),
                nn.BatchNorm2d(outp),
            )

        layers = []
        layers.append(block(inp, outp, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(outp, outp))
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class ConvBN(nn.Module):
    def __init__(self, inp, oup):
        super(ConvBN, self).__init__()
        self.conv_lyr = nn.Conv2d(inp, oup, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv_lyr(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

    
class Net(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(Net, self).__init__()
        self.first_lyr = ConvBN(3, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResLayer(SEBottleneck, 64, 3)
        self.layer2 = ResLayer(SEBottleneck, 128, 4, stride=2)
        self.layer3 = ResLayer(SEBottleneck, 256, 6, stride=2)
        self.layer4 = ResLayer(SEBottleneck, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        out = self.first_lyr(x)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
        