
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleBlock_3(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C/g,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)


class Bottleneck_3(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes/4
        g = 1 if in_planes==24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock_3(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out

class shuffle_layer_3(nn.Module):
    def __init__(self,out_planes, num_blocks, groups):
        super(shuffle_layer,self).__init__()
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck_3(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        out = self.layers(x)
        return out 
class shuffle_basic_block_3(nn.Module):
    def __init__(self):
        super(shuffle_basic_block_3, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class ShuffleNet(nn.Module):
    def __init__(self):
        super(ShuffleNet, self).__init__()
        self.basic_block_3 = shuffle_basic_block_3()
        self.layer1_3 = shuffle_layer_3(200, 4, 2)
        self.layer2_3 = shuffle_layer_3(400, 8, 2)
        self.layer3_3 = shuffle_layer_3(800, 4, 2)
        self.linear_3 = nn.Linear(400, 10)

    def forward(self, x):
        out = self.basic_block_33(x)
        out = self.layer1_3(out)
        out = self.layer2_3(out)
        out = self.layer3_3(out)
        out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)
        out = self.linear_3(out)
        return out
