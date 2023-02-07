from torch import nn

from utils.tools import *


class GT_Module(nn.Module):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        super(GT_Module, self).__init__()
        if ("Push" in env_config.name) or ("Maze" in env_config.name):
            self.num_slots = env_config.num_objects_range[1] + 2
        else:
            self.num_slots = env_config.num_objects_range[1] + 1
        self.rep_dim = env_config.state_size
        mlp = []
        in_dim = self.rep_dim
        for dim, act in zip(ocr_config.dims, ocr_config.acts):
                mlp.append(nn.Linear(in_dim, dim))
                if act == "relu":
                    mlp.append(nn.ReLU())
                in_dim = dim
        self._net = nn.Sequential(*mlp)
        if len(ocr_config.dims) > 0:
            self.rep_dim = ocr_config.dims[-1]

    def forward(self, obs: Tensor) -> Tensor:
        #return obs
        return self._net(obs)

    def get_loss(self, obs: Tensor, with_rep=False) -> dict:
        if with_rep:
            #return {}, obs
            return {}, self._net(obs)
        else:
            return {}

    def get_samples(self, obs: Tensor) -> dict:
        return {}
