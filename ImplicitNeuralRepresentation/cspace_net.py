import torch
import torch.nn as nn
import numpy as np

class PositionEmbedder(nn.Module):
    def __init__(self, dof=7, num_freq=16, sigma=3):
        super(PositionEmbedder, self).__init__()

        self.dof = dof
        self.num_freq = num_freq
        self.sigma = sigma

        self.freq = nn.Linear(in_features=3, out_features=self.num_freq)
        with torch.no_grad(): # fix these weights
            self.freq.weight = nn.Parameter(torch.normal(mean=0, std=self.sigma, size=(self.num_freq, self.dof)), requires_grad=False)
            self.freq.bias = nn.Parameter(torch.zeros(self.num_freq), requires_grad=False)

        return
    
    def to(self, device):
        on_device_model = super().to(device)
        return on_device_model
    
    """
    Input is (self.dof)-dimensional
    Output is (2*self.num_freq)-dimensional
    # Uses Fourier Features: https://bmild.github.io/fourfeat/
    """
    def forward(self, x):
        #assert tuple(x[0].shape) == (3,)
        
        # sin & cosine
        x = self.freq(x)
        x = torch.cat([torch.sin(2 * np.pi * x), torch.cos(2 * np.pi * x)], 
                    dim=1)
        # assert len(x.shape) == 2
        # assert x.shape[1] == 2 * self.num_freq

        return x

class CSpaceNet(nn.Module):
    def __init__(self, dof, num_freq, sigma):
        super(CSpaceNet, self).__init__()

        self.dof = dof
        self.num_freq = num_freq
        self.sigma = sigma

        self.position_embedder = PositionEmbedder(dof=self.dof, num_freq=self.num_freq, sigma=self.sigma)

        self.block1 = nn.Sequential(
            nn.Linear(2*self.num_freq, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(2*self.num_freq + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def to(self, device):
        on_device_net = super().to(device)
        on_device_net.position_embedder = self.position_embedder.to(device)
        return on_device_net

    def forward(self, x):
        positional_encoding = self.position_embedder(x)
        x_intermediate = self.block1(positional_encoding)
        x_output = self.block2(torch.cat((positional_encoding, x_intermediate), dim=1))
        return x_output