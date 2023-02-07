import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import *


# Taken from https://github.com/addtt/object-centric-library/blob/main/models/shared/nn.py#L45-L67
class PositionalEmbedding(nn.Module):
    def __init__(self, obs_size: int, obs_channels: int):
        super().__init__()
        width = height = obs_size
        east = torch.linspace(0, 1, width).repeat(height)
        west = torch.linspace(1, 0, width).repeat(height)
        south = torch.linspace(0, 1, height).repeat(width)
        north = torch.linspace(1, 0, height).repeat(width)
        east = east.reshape(height, width)
        west = west.reshape(height, width)
        south = south.reshape(width, height).T
        north = north.reshape(width, height).T
        # (4, h, w)
        linear_pos_embedding = torch.stack([north, south, west, east], dim=0)
        linear_pos_embedding.unsqueeze_(0)  # for batch size
        self.channels_map = nn.Conv2d(4, obs_channels, kernel_size=1)
        self.register_buffer("linear_position_embedding", linear_pos_embedding)

    def forward(self, x: Tensor) -> Tensor:
        bs_linear_position_embedding = self.linear_position_embedding.expand(
            x.size(0), 4, x.size(2), x.size(3)
        )
        x = x + self.channels_map(bs_linear_position_embedding)
        return x


# Taken from https://github.com/singhgautam/slate/blob/master/train.py#L131-L146
def cosine_anneal(step, start_value, final_value, start_step, final_step):
    assert start_value >= final_value
    assert start_step <= final_step
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b
    return value


# Taken from https://github.com/singhgautam/slate/blob/master/train.py#L113-L128
def linear_warmup(step, start_value, final_value, start_step, final_step):
    assert start_value <= final_value
    assert start_step <= final_step
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b
    return value


def gumbel_max(logits, dim=-1):
    eps = torch.finfo(logits.dtype).tiny
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = logits + gumbels
    return gumbels.argmax(dim)


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    eps = torch.finfo(logits.dtype).tiny
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim)
    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def log_prob_gaussian(value, mean, std):
    var = std**2
    if isinstance(var, float):
        return -0.5 * (
            ((value - mean) ** 2) / var + math.log(var) + math.log(2 * math.pi)
        )
    else:
        return -0.5 * (((value - mean) ** 2) / var + var.log() + math.log(2 * math.pi))
