# Taken from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L51-L93
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from utils.tools import *


class NatureCNN(nn.Module):
    def __init__(self, in_dim, rep_dim, use_cnn_feat):
        super(NatureCNN, self).__init__()

        net = [
            nn.Conv2d(in_dim, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        ]
        if not use_cnn_feat:
            net.append(nn.Flatten())
            net.append(nn.Linear(1024, rep_dim))
            net.append(nn.ReLU())
        self._net = nn.Sequential(*net)

    def forward(self, inp: Tensor) -> Tensor:
        return self._net(inp)
