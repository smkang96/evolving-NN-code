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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

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
        self.layer1 = ResLayer(BasicBlock, 64, 3)
        self.layer2 = ResLayer(BasicBlock, 128, 4, stride=2)
        self.layer3 = ResLayer(BasicBlock, 256, 6, stride=2)
        self.layer4 = ResLayer(BasicBlock, 512, 3, stride=2)
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
        