import torch
from torch import nn
from itertools import permutations

from utils.tools import *


class RN_Module(nn.Module):
    def __init__(self, ocr_rep_dim: int, ocr_num_slots: int, num_stacked_obss: int, config: dict) -> None:
        super(RN_Module, self).__init__()
        assert num_stacked_obss == 1, "Not implemented yet"
        #TODO(yifu): take into account num_stacked_obss
        self.rep_dim = config.f_dims[-1]
        # g network
        g_dims = config.g_dims
        g_net = []
        in_dim = ocr_rep_dim * 2
        for _dim in g_dims:
            g_net.append(nn.Linear(in_dim, _dim))
            g_net.append(nn.ReLU())
            in_dim = _dim
        self._g = nn.Sequential(*g_net)
        # f network
        f_dims = config.f_dims
        f_net = []
        in_dim = g_dims[-1]
        for _dim in f_dims:
            f_net.append(nn.Linear(in_dim, _dim))
            f_net.append(nn.ReLU())
            in_dim = _dim
        self._f = nn.Sequential(*f_net)

    def _get_permutations(self, state):
        B, S, D = state.shape
        paired_state = []
        idx_list = list(range(state.shape[1]))
        flag = True
        for i, j in permutations(idx_list, 2):
            if flag:
                paired_state = torch.cat([state[:, i], state[:, j]], dim=-1)
                flag = False
            else:
                paired_state = torch.cat(
                    [paired_state, torch.cat([state[:, i], state[:, j]], dim=-1)], dim=1
                )
        paired_state = paired_state.reshape(B, -1, D * 2)
        return paired_state

    def forward(self, state: Tensor) -> Tensor:
        # [batch_size, num_slots, rep_dim] -> [batch_size, d_model]
        B, S, D = state.shape
        # [batch_size, num_slots P 2, rep_dim*2]
        state = self._get_permutations(state)
        B, S, D = state.shape
        state = self._g(state.reshape(-1, D))
        # [batch_size, d_model]
        state = state.reshape(B, S, -1).sum(dim=1)
        # [batch_size, f_dims[-1]]
        return self._f(state)
