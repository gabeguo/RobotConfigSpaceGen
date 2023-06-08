import torch
import torch.nn as nn
import numpy as np

class CSpaceNet(nn.Module):
    def __init__(self, dof, num_freq, sigma):
        super(CSpaceNet, self).__init__()

        self.dof = dof
        self.num_freq = num_freq
        self.sigma = sigma

        self.block1 = nn.Sequential(
            nn.Linear(self.dof, 256),
            #nn.Linear(self.num_freq * 2 * self.dof, 256),
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
            nn.Linear(self.dof + 256, 256),
            #nn.Linear(self.num_freq * 2 * self.dof + 256, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.BatchNorm1d(num_features=1),
        )

    def to(self, device):
        on_device_net = super().to(device)
        on_device_net.position_embedder = self.position_embedder.to(device)
        return on_device_net

    def forward(self, x):
        # positional_encoding = self.position_embedder(x)
        # x_intermediate = self.block1(positional_encoding)
        # x_output = self.block2(torch.cat((positional_encoding, x_intermediate), dim=1))
        # return x_output
    
        x_intermediate = self.block1(x)
        x_output = self.block2(torch.cat((x_intermediate, x), dim=1))
        return x_output