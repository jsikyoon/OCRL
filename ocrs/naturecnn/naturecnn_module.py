# Doesn't support aux loss yet
# Taken from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L51-L93
import torch
from torch import nn
from torch.nn import functional as F

from utils.tools import *
from ocrs.base import Base


class NatureCNN_Module(nn.Module):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        super(NatureCNN_Module, self).__init__()
        obs_size = env_config.obs_size
        obs_channels = env_config.obs_channels
        self.rep_dim = ocr_config.rep_dim
        self._use_cnn_feat = ocr_config.use_cnn_feat

        if self._use_cnn_feat:
            if ocr_config.cnn_feat_size == 4:
                self.rep_dim = 64
                self.num_slots = 4**2
            elif ocr_config.cnn_feat_size == 2:
                self.rep_dim = 128
                self.num_slots = 2**2
        else:
            self.rep_dim = ocr_config.rep_dim
            self.num_slots = 1

        cnn = [
            nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        ]
        if ocr_config.cnn_feat_size == 2:
            cnn.append(nn.Conv2d(64,128, 3, 1, 0))
            cnn.append(nn.ReLU())
        if not self._use_cnn_feat:
            cnn.append(nn.Flatten())
        self._cnn = nn.Sequential(*cnn)
        if not self._use_cnn_feat:
            # Compute shape by doing one forward pass
            with torch.no_grad():
                n_flatten = self._cnn(
                    torch.zeros(1, obs_channels, obs_size, obs_size)
                ).shape[1]
            self._linear = nn.Sequential(nn.Linear(n_flatten, self.rep_dim), nn.ReLU())

    def _get_cnn_featuremap(self, obs: Tensor) -> Tensor:
        rep = self._cnn(obs)
        rep = rep.permute(0, 2, 3, 1)
        return rep.reshape(rep.shape[0], -1, rep.shape[-1])

    def forward(self, obs: Tensor) -> Tensor:
        """
        obs: Batch_size, dim
        """
        if self._use_cnn_feat:
            return self._get_cnn_featuremap(obs)
        else:
            return self._linear(self._cnn(obs))

    def get_loss(self, obs: Tensor, with_rep=False) -> dict:
        if with_rep:
            if self._use_cnn_feat:
                rep = self._get_cnn_featuremap(obs)
                return {}, rep
            else:
                return {}, self._nets["linear"](self._nets["cnn"](obs))
        else:
            return {}
