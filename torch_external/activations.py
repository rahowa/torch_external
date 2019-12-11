import torch 
import torch.nn as nn 
import torch.nn.functional as F


class Mish(nn.Module):    
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class HardSwish(nn.Module):
    def __init__(self):
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return x * self.relu(x + 3) / 6.0
