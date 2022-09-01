import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class lossAV(nn.Module):
	def __init__(self):
		super(lossAV, self).__init__()
		self.criterion = nn.CrossEntropyLoss()
		self.FC        = nn.Linear(256, 2)
		
	def forward(self, x, labels=None):	
		in_feature = x.squeeze(1)
		in_feature = self.FC(in_feature)
		if labels == None:
			predScore = in_feature[:,1]
			predScore = predScore.t()
			predScore = predScore.view(-1).detach().cpu().numpy()
			return predScore
		else:
			nloss = self.criterion(in_feature, labels)
			predScore = F.softmax(in_feature, dim = -1)
			predLabel = torch.round(F.softmax(in_feature, dim = -1))[:,1]
			correctNum = (predLabel == labels).sum().float()
			return nloss, predScore, predLabel, correctNum
			# return nloss, in_feature, predLabel, correctNum

class lossA(nn.Module):
	def __init__(self):
		super(lossA, self).__init__()
		self.criterion = nn.CrossEntropyLoss()
		self.FC        = nn.Linear(128, 2)

	def forward(self, x, labels):	
		x = x.squeeze(1)
		x = self.FC(x)	
		nloss = self.criterion(x, labels)
		return nloss

class lossV(nn.Module):
	def __init__(self):
		super(lossV, self).__init__()

		self.criterion = nn.CrossEntropyLoss()
		self.FC        = nn.Linear(128, 2)

	def forward(self, x, labels):	
		x = x.squeeze(1)
		x = self.FC(x)
		nloss = self.criterion(x, labels)
		return nloss

