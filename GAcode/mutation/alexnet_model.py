import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self):
		super(NET, self).__init__()
		self.layer1 = AlexBlock(3, 64, 11, 4, 2)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
		self.layer2 = AlexBlock(64, 192, 5, padding=2)
		self.layer3 = AlexBlock(192, 384, 3, padding=1)
		self.layer4 = AlexBlock(384, 256, 3, padding=1)
		self.layer5 = AlexBlock(256, 256, 3, padding=1)
		# forward

	def forward(self, x):
		out = self.layer1(x)
		out = self.maxpool(out)
		out = self.layer2(out)
		out = self.maxpool(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		out = self.maxpool(out)
		#return out 