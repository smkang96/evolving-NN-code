'''translation of NN code to samplable format: VGG19'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, inp, outp, lnum):
        super(VGGBlock, self).__init__()
        self.first_lyr = nn.Sequential(
            nn.Conv2d(inp, outp, kernel_size=3, padding=1), 
            nn.BatchNorm2d(outp), 
            nn.ReLU(inplace=True)
        )
        leftover_lyrs = []
        for li in range(1, lnum):
            leftover_lyrs.append(nn.Conv2d(outp, outp, kernel_size=3, padding=1))
            leftover_lyrs.append(nn.BatchNorm2d(outp)) # uses BN as default
            leftover_lyrs.append(nn.ReLU(inplace=True))
        self.next_lyrs = nn.Sequential(*leftover_lyrs)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.first_lyr(x)
        out = self.next_lyrs(out)
        out = self.maxpool(out)
        return out
    
class VGGClassifier(nn.Module):
    def __init__(self, inp, class_num):
        super(VGGClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(inp, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = VGGBlock(3, 64, 2)
        self.layer2 = VGGBlock(64, 128, 2)
        self.layer3 = VGGBlock(128, 256, 4)
        self.layer4 = VGGBlock(256, 512, 4)
        self.layer5 = VGGBlock(512, 512, 4)
        self.vggc = VGGClassifier(512 * 1 * 1, 10) # 10 should be immutable...
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.vggc(out) # returns 2d
        return out
        