import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexBlock(nn.Module):
	def __init__(self, inp, outp, k_size, stride=1, padding):
		self.block = nn.Sequential(
			nn.Conv2d(inp, outp, kernel_size = k_size, stride = stride, padding=padding),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		out = self.block(x)
		return x

class AlexClassifier(nn.Module):
	def __init__(self, num_classes=1000):
		self.classifier = nn.Sequential(
			nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.layer1 = AlexBlock(3, 64, 11, 4, 2)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
		self.layer2 = AlexBlock(64, 192, 5, padding=2)
		self.layer3 = AlexBlock(192, 384, 3, padding=1)
		self.layer4 = AlexBlock(384, 256, 3, padding=1)
		self.layer5 = AlexBlock(256, 256, 3, padding=1)
		#number of classes is 1000?
		self.classifier = AlexClassifier(1000)

	def forward(self, x):
		out = self.layer1(x)
		out = self.maxpool(out)
		out = self.layer2(out)
		out = self.maxpool(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		out = self.maxpool(out)
		out = self.classifier(out)
		return out