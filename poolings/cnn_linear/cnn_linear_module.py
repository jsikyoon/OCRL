from torch import nn

from utils.tools import *
from poolings.common.naturecnn import NatureCNN


class CNN_Linear_Module(nn.Module):
    def __init__(self, ocr_rep_dim: int, ocr_num_slots: int, config: dict, num_stacked_obss: int=1) -> None:
        super(CNN_Linear_Module, self).__init__()
        self.rep_dim = rep_dim = config.rep_dim
        self._net = NatureCNN(ocr_rep_dim * num_stacked_obss, rep_dim, use_cnn_feat=False)

    def forward(self, state: Tensor) -> Tensor:
        return self._net(slot_to_img(state))
