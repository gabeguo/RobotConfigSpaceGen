import torch
import torch.nn as nn
import numpy as np

class PositionEmbedder(nn.Module):
    def __init__(self, dof=7, num_freq=10, sigma=1):
        super(PositionEmbedder, self).__init__()

        self.dof = dof
        self.num_freq = num_freq
        self.sigma = sigma

        return
    
    def to(self, device):
        on_device_model = super().to(device)
        return on_device_model
    
    """
    Input is self.dof-dimensional
    Output is 2*3*self.dof-dimensional
    # Uses Fourier Features: https://bmild.github.io/fourfeat/
    """
    def forward(self, x):
        x.shape[1] == self.dof

        # sin & cosine
        x = torch.cat([torch.sin(i * self.sigma * x) for i in range(1, self.num_freq + 1)] + \
                    [torch.cos(i * self.sigma * x) for i in range(1, self.num_freq + 1)], 
                    dim=1)
        assert len(x.shape) == 2
        assert x.shape[1] == 2 * self.dof * self.num_freq

        return x

class CSpaceNet(nn.Module):
    def __init__(self, dof, num_freq, sigma):
        super(CSpaceNet, self).__init__()

        self.dof = dof
        self.num_freq = num_freq
        self.sigma = sigma

        self.position_embedder = PositionEmbedder(dof=self.dof, num_freq=self.num_freq, sigma=self.sigma)

        self.block1 = nn.Sequential(
            #nn.Linear(self.dof, 256),
            nn.Linear(self.num_freq * 2 * self.dof, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            #nn.Linear(self.dof + 256, 256),
            nn.Linear(self.num_freq * 2 * self.dof + 256, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.BatchNorm1d(num_features=1),
        )

        return

    def to(self, device):
        on_device_net = super().to(device)
        on_device_net.position_embedder = self.position_embedder.to(device)
        return on_device_net

    def forward(self, x):
        positional_encoding = self.position_embedder(x)
        x_intermediate = self.block1(positional_encoding)
        x_output = self.block2(torch.cat((positional_encoding, x_intermediate), dim=1))
        return x_output
    
        # x_intermediate = self.block1(x)
        # x_output = self.block2(torch.cat((x_intermediate, x), dim=1))
        # return x_output