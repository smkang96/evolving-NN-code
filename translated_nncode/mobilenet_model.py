'''translation of NN code to samplable format: MobileNet'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBN_mobilenet(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ConvBN_mobilenet, self).__init__()
        self.conv_lyr = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv_lyr(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ConvDW_mobilenet(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ConvDW_mobilenet, self).__init__()
        self.conv_lyr1 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        # nn.ReLU(inplace=True),

        self.conv_lyr2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        # nn.ReLU(inplace=True),
    
    def forward(self, x):
        out = self.conv_lyr1(x)
        out = F.relu(self.bn1(out))
        out = self.conv_lyr2(out)
        out = F.relu(self.bn2(out))
        return out
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.layerM_01 = ConvBN_mobilenet(3, 32, 2)
        self.layerM_02 = ConvDW_mobilenet(32, 64, 1)
        self.layerM_03 = ConvDW_mobilenet(64, 128, 1)
        self.layerM_04 = ConvDW_mobilenet(128, 128, 1)
        self.layerM_05 = ConvDW_mobilenet(128, 256, 2)
        self.layerM_06 = ConvDW_mobilenet(256, 256, 1)
        self.layerM_07 = ConvDW_mobilenet(256, 512, 2)
        self.layerM_08 = ConvDW_mobilenet(512, 512, 1)
        self.layerM_09 = ConvDW_mobilenet(512, 512, 1)
        self.layerM_10 = ConvDW_mobilenet(512, 512, 1)
        self.layerM_11 = ConvDW_mobilenet(512, 512, 1)
        self.layerM_12 = ConvDW_mobilenet(512, 512, 1)
        self.layerM_13 = ConvDW_mobilenet(512, 1024, 2)
        self.layerM_14 = ConvDW_mobilenet(1024, 1024, 1)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.layerM_01(x)
        out = self.layerM_02(out)
        out = self.layerM_03(out)
        out = self.layerM_04(out)
        out = self.layerM_05(out)
        out = self.layerM_06(out)
        out = self.layerM_07(out)
        out = self.layerM_08(out)
        out = self.layerM_09(out)
        out = self.layerM_10(out)
        out = self.layerM_11(out)
        out = self.layerM_12(out)
        out = self.layerM_13(out)
        out = self.layerM_14(out)
        out = self.avgpool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out
