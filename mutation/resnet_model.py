'''translation of NN code to samplable format: ResNet34'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.first_lyr = ConvBN(3, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResLayer(64, 64, 3)
        self.layer2 = ResLayer(64, 128, 4, stride=2)
        self.layer3 = ResLayer(128, 256, 6, stride=2)
        self.layer4 = ResLayer(256, 512, 3, stride=2)
# forward     
    def forward(self, x):
        out = self.first_lyr(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
# return out 
        