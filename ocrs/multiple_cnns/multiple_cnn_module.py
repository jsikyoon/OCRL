# Doesn't support aux loss yet
# Taken from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L51-L93
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf, open_dict

from utils.tools import *
from ocrs.base import Base
from ocrs.naturecnn import NatureCNN_Module


class MultipleCNN_Module(nn.Module):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        super(MultipleCNN_Module, self).__init__()
        obs_size = env_config.obs_size
        obs_channels = env_config.obs_channels
        self.rep_dim = ocr_config.rep_dim
        self.num_slots = ocr_config.num_modules

        with open_dict(ocr_config):
            ocr_config.cnn_feat_size = 4
            ocr_config.use_cnn_feat = False
        self._cnns = nn.ModuleList(
                [NatureCNN_Module(ocr_config, env_config)
                    for i in range(self.num_slots)])

    def forward(self, obs: Tensor) -> Tensor:
        """
        obs: Batch_size, dim
        """
        reps = []
        for i in range(self.num_slots):
            reps.append(self._cnns[i](obs))
        # [batch_size, num_slots, dim]
        reps = torch.stack(reps, axis=1)
        return reps

    def get_loss(self, obs: Tensor, with_rep=False) -> dict:
        if with_rep:
            return {}, self(obs)
        else:
            return {}
