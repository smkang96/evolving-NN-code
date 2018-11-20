import torch.nn as nn

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1_5_0 = nn.Conv2d(3, 24, kernel_size=3, padding=1,bias=False)
		self.dense1_5_1 = make_dense_5(24,'+192', 12, 16, True)
		self.trans1_5_2 = Transition_5(216, 108)
		self.dense2_5_3 = make_dense_5(108,'+192',12,16,True)
#forward
	def forward(self, x):
		out = self.conv1_5_0(x)
		out = self.dense1_5_1(out)
		out = self.trans1_5_2(out)
		out = self.dense2_5_3(out)
#return