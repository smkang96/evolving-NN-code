import torch.nn as nn
import torch.nn.functional as F

class HighwayBlock_9(nn.Module):
	def __init__(self, size, f=F.softmax):
		super(Highway, self).__init__()


		self.nonlinear = nn.ModuleList([nn.Linear(size, size)])

		self.linear = nn.ModuleList([nn.Linear(size, size)])

		self.gate = nn.ModuleList([nn.Linear(size, size)])

		self.f = f

	def forward(self, x):
		
		gate = F.sigmoid(self.gate[layer](x))
		nonlinear = self.f(self.nonlinear[layer](x))
		linear = self.linear[layer](x)

		x = gate * nonlinear + (1 - gate) * linear

		return x

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.layer1_9 = HighwayBlock(64)
	def forward(self, x):
		out = self.layer1_9(x)
		out = self.layer1_9(x)
		out = self.layer1_9(x)
		out = self.layer1_9(x)
		out = self.layer1_9(x)
		out = self.layer1_9(x)
		out = self.layer1_9(x)
		out = self.layer1_9(x)
		out = self.layer1_9(x)
		out = self.layer1_9(x)
		return out
