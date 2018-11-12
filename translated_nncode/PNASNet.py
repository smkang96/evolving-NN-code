
import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv_6(nn.Module):
    '''Separable Convolution.'''
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(SepConv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size, stride,
                               padding=(kernel_size-1)//2,
                               bias=False, groups=in_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn1(self.conv1(x))


class CellA_6(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(CellA, self).__init__()
        self.stride = stride
        self.sep_conv1 = SepConv_6(in_planes, out_planes, kernel_size=7, stride=stride)
        if stride==2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            y2 = self.bn1(self.conv1(y2))
        return F.relu(y1+y2)


class PNAS_make_layer_6(nn.Module):
    def __init__(self, planes, num_cells):
        super(PNAS_make_layer_6, self).__init__()
        layers = []
        for _ in range(num_cells):
            layers.append(CellA_6(44, planes, stride=1))
            self.in_planes = planes
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        out = self.layers(x)
        return out

class PNAS_downsample_6(nn.Module):
    def __init__(self, planes):
        super(PNAS_downsample_6,self).__init__()
        layer = CellA_6(44, planes, stride=2)
        self.layer = layer 
    def forward(self, x):
        out = self.layer(x)
        return out
class PNAS_basic_block_6(nn.Module):
    def __init__(self):
        super (PNAS_basic_block_6,self).__init__()
        self.conv1 = nn.Conv2d(3, 44, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(44)
    def forward(self, x):
        out = out = F.relu(self.bn1(self.conv1(x)))
        return out


class PNASNet(nn.Module):
    def __init__(self):
        super(PNASNet, self).__init__()

        self.basic_block_6 = PNAS_basic_block_6()
        self.layer1_6 = PNAS_make_layer_6(44, num_cells=6)
        self.layer2_6 = PNAS_downsample_6(44*2)
        self.layer3_6 = PNAS_make_layer_6(44*2, num_cells=6)
        self.layer4_6 = PNAS_downsample_6(44*4)
        self.layer5_6 = PNAS_make_layer_6(44*4, num_cells=6)
        self.linear_6 = nn.Linear(44*4, 10)


    def forward(self, x):
        out = self.basic_block_6(x)
        out = self.layer1_6(out)
        out = self.layer2_6(out)
        out = self.layer3_6(out)
        out = self.layer4_6(out)
        out = self.layer5_6(out)
        out = F.avg_pool2d(out, 8)
        #out = out.view(out.size(0), -1)
        out = self.linear_6(out)
        return out
