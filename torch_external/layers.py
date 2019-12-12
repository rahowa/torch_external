import torch 
import torch.nn as nn 
import torch.nn.functional as F


__all__ = ["FilterResponseNorm1D", "FilterResponseNorm2D",
           "FilterResponseNorm3D", "Squash", "Flatten", "AttentionGates"]


class FilterResponseNorm1D(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.channels = channels
        self.tau = nn.Parameter(torch.zeros(channels, 1))
        self.gamma = nn.Parameter(torch.ones(channels, 1))
        self.beta = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x):
        nu2 = (x**2).mean(2, keepdim=True)
        x = x * torch.rsqrt_(nu2 + self.eps)
        return torch.max([self.gamma * x + self.beta, self.tau])


class FilterResponseNorm2D(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.channels = channels
        self.tau = nn.Parameter(torch.zeros(channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x):
        nu2 = (x**2).mean((2, 3), keepdim=True)
        x = x * torch.rsqrt_(nu2 + self.eps)
        return torch.max([self.gamma * x + self.beta, self.tau])


class FilterResponseNorm3D(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps 
        self.channels = channels
        self.tau = nn.Parameter(torch.zeros(channels, 1, 1, 1))
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(channels, 1, 1, 1))

    def forward(self, x):
        nu2 = (x**2).mean((2, 3, 4), keepdim=True)
        x = x * torch.rsqrt_(nu2 + self.eps)
        return torch.max([self.gamma * x + self.beta, self.tau])


class Squash1D(nn.Module):
    """
    BSxSEQLENxFEATURES -> BSxFEATURES
    """
    def __init__(self, method):
        super().__init__()
        self.method = method
        
    def forward(self, x):
        if self.method == "last":
            return x[:, -1, :].view(-1, x.size(-1))
        elif self.method == "flatten":
            return x.reshape(-1, x.size(1) * x.size(2))
        elif self.method == "max":
            return torch.max(x, 1)[0].view(-1, x.size(-1))
        elif self.method == "avg":
            return torch.mean(x, 1).view(-1, x.size(-1))
        else:
            raise NotImplementedError


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        bs = x.size(0)
        return x.view(bs, -1)


class AttentionGates(nn.Module):
    """ Base implementation of GRID ATTENTION LAYER
        based on https://openreview.net/pdf?id=Skft7cijM
        
        Full implementation at 
        https://github.com/ozan-oktay/Attention-Gated-Networks
    """
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels
        self.sub_sample_factor = (2, 2, 2)
        
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        self.theta = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1,#self.sub_sample_factor, 
                               stride=2,#self.sub_sample_factor, 
                               padding=0,
                               bias=False)
        self.phi = nn.Conv2d(in_channels=self.gating_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1, 
                             stride=1, 
                             padding=0, 
                             bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels,
                             out_channels=1, 
                             kernel_size=1, 
                             stride=1, 
                             padding=0, 
                             bias=True)
        
    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))

        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='bilinear')
        y = sigm_psi_f.expand_as(x) * x
        return y

    def forward(self, x, g):
        output = self._concatenation(x, g)
        return output



    