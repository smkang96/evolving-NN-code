'''translation of NN code to samplable format: VGG19'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.layer1 = VGGBlock(3, 64, 2)
        self.layer2 = VGGBlock(64, 128, 2)
        self.layer3 = VGGBlock(128, 256, 4)
        self.layer4 = VGGBlock(256, 512, 4)
        self.layer5 = VGGBlock(512, 512, 4)
        #forward 
        self.avgpool_end = nn.AvgPool2d(8, stride=1)
        self.linear_end = nn.Linear(176,10)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        #return        out = self.avgpool_end(out)
        out = out.view(out.size(0), -1)
        out = self.linear_end(out)
        return out
