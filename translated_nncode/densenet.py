import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class Bottleneck_5(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer_5(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition_5(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class make_dense_5(nn.Module):
    def __init__(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        super(_make_dense_5,self).__init__()
        self.nChannels = nChannels
        self.growthRate = growthRate
        self.nDenseBlocks = nDenseBlocks
        self.bottleneck = bottleneck

        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        self.nn.Sequential(*layers)
        #return nn.Sequential(*layers)
    def forward(self,x):
        out = self.layers(x)
        return out



class DenseNet(nn.Module):
    def __init__(): 
        super(DenseNet, self).__init__()
        nChannels = 2*growthRate
        #nChannels = 24
        self.conv1_5 = nn.Conv2d(3, 24, kernel_size=3, padding=1,bias=False)
        self.dense1_5 = _make_dense_5(24, 12, 16, True)
        # nChannels += nDenseBlocks*growthRate
        # nChannels = 216
        # nOutChannels = int(math.floor(nChannels*reduction))
        # nOutChannels = 108
        self.trans1_5 = Transition_5(216, 108)

        # nChannels = nOutChannels
        #nChannels = 108
        self.dense2_5 = _make_dense_5(108, 12, 16, True)
        # nChannels += nDenseBlocks*growthRate
        # nChannels = 300
        # nOutChannels = int(math.floor(nChannels*reduction))
        # nOutChannels = 150
        self.trans2_5 = Transition_5(300, 150)

        # nChannels = nOutChannels
        # nChannels = 150
        self.dense3_5 = _make_dense_5(150, 12, 16, True)
        #nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(342)
        self.fc = nn.Linear(342, 10)



    def forward(self, x):
        out = self.conv1_5(x)
        out = self.trans1_5(self.dense1(out))
        out = self.trans2_5(self.dense2(out))
        out = self.dense3_5(out)
        out = F.relu(self.bn1(out))
        out = torch.squeeze(F.avg_pool2d(out,8))
        out = F.log_softmax(self.fc(out))
        return out