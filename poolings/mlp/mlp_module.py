from torch import nn

from utils.tools import *


class MLP_Module(nn.Module):
    def __init__(self, ocr_rep_dim: int, ocr_num_slots: int, config: dict, num_stacked_obss: int=1) -> None:
        super(MLP_Module, self).__init__()
        self.rep_dim = config.dims[-1]
        in_dim = ocr_rep_dim * ocr_num_slots * num_stacked_obss

        # MLP
        net = []
        for dim, act in zip(config.dims, config.acts):
            net.append(nn.Linear(in_dim, dim))
            if act == "relu":
                net.append(nn.ReLU())
            in_dim = dim
        self._mlp = nn.Sequential(*net)

    def forward(self, state: Tensor) -> Tensor:
        state = state.flatten(start_dim=1) if len(state.shape) == 3 else state
        return self._mlp(state)
