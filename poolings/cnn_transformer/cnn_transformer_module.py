from torch import nn

from utils.tools import *
from poolings.common.naturecnn import NatureCNN
from poolings.common.transformer import (
    Transformer,
    LearnedPositionalEncoding,
    PositionalEncoding,
)


class CNN_Transformer_Module(nn.Module):
    def __init__(self, ocr_rep_dim: int, ocr_num_slots: int, config: dict, num_stacked_obss: int=1) -> None:
        super(CNN_Transformer_Module, self).__init__()
        self.rep_dim = d_model = config.d_model

        # CNN
        self._cnn = NatureCNN(ocr_rep_dim, rep_dim=None, use_cnn_feat=True)

        # Transformer
        nhead = config.nhead
        num_layers = config.num_layers
        # Positional encoding
        if config.pos_emb == "ape":  # Absolute Positional Embedding
            pos = PositionalEncoding(d_model=d_model, max_len=ocr_num_slots + 1)
        elif config.pos_emb == "lpe":  # Learnable Positional Embedding
            pos = PositionalEncoding(d_model=d_model, max_len=ocr_num_slots + 1)
        elif config.pos_emb == "None":
            pos = None
        self._trans = Transformer(64, d_model, nhead, num_layers, pos)

    def forward(self, state: Tensor) -> Tensor:
        # [B, D, H, W]
        state = self._cnn(slot_to_img(state))
        # [B, H, W, D]
        state = state.permute(0, 2, 3, 1)
        # [B, N, D]
        state = state.reshape(state.shape[0], -1, state.shape[-1])
        # [B, D]
        return self._trans(state)
