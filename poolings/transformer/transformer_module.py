from torch import nn

from utils.tools import *
from poolings.common.transformer import (
    Transformer,
    LearnedPositionalEncoding,
    PositionalEncoding,
    StackedObsPositionalEncoding,
)

class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalEncoding, self).__init__()
        se = torch.zeros(max_len + 1, d_model)
        inp = torch.arange(0, max_len + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        se[:, 0::2] = torch.sin(inp * div_term)
        se[:, 1::2] = torch.cos(inp * div_term)

        se = se.unsqueeze(0).transpose(0, 1)
        self.register_buffer('se', se)

    def forward(self, x):
        return self.se[x, :]


class Transformer_Module(nn.Module):
    def __init__(self, ocr_rep_dim: int, ocr_num_slots: int, config: dict, num_stacked_obss: int=1) -> None:
        super(Transformer_Module, self).__init__()
        self.rep_dim = d_model = config.d_model
        self.config = config
        nhead = config.nhead
        num_layers = config.num_layers
        norm_first = config.norm_first

        # Positional encoding
        if num_stacked_obss > 1:
            pos = StackedObsPositionalEncoding(d_model=d_model, max_len=(ocr_num_slots*num_stacked_obss) + 1, num_stacked_obss=num_stacked_obss, include_initial_cls_token=True)
        elif config.pos_emb == "ape":  # Absolute Positional Embedding
            pos = PositionalEncoding(d_model=d_model, max_len=ocr_num_slots + 1)
        elif config.pos_emb == "lpe":  # Learnable Positional Embedding
            pos = PositionalEncoding(d_model=d_model, max_len=ocr_num_slots + 1)
        elif config.pos_emb == "None":
            pos = None

        if config.use_mlp1:
            self.mlp = nn.Sequential(
                nn.Linear(ocr_rep_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
            )
            ocr_rep_dim = 128

        if config.use_mlp2:
            self.mlp = nn.Sequential(
                nn.Linear(ocr_rep_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
            )
            ocr_rep_dim = 128

        if config.cw_embedding:
            self.max_len = 10000
            self.cw_emb = SinusoidalEncoding(d_model=d_model, max_len=self.max_len)
            self.arm_emb = nn.Linear(28 * d_model, 128)
            self.obj_emb = nn.Linear((3 * d_model) + 3, 128)
            ocr_rep_dim = 128

        if config.push_embedding:
            self.max_len = 10000000
            self.color_emb = nn.Embedding(10, 128)
            self.shape_emb = nn.Embedding(10, 128)
            self.pos_emb = SinusoidalEncoding(d_model=d_model, max_len=self.max_len)
            self.obj_emb = nn.Linear((4 * d_model), 128)
            ocr_rep_dim = 128

        self._trans = Transformer(ocr_rep_dim, d_model, nhead, num_layers, pos, norm_first)

    def get_pos_emb(self, emb, x):
        x = (x + 1) / 2
        x = torch.clamp(x, 0.0, 1.0)
        x = (x // (1 / self.max_len)).long()
        x = emb(x)[..., -1, :]
        return x

    def forward(self, state: Tensor) -> Tensor:
        if self.config.push_embedding:
            color = self.color_emb(state[:, :, 0].long())
            shape = self.shape_emb(state[:, :, 1].long())
            pos = state[:, :, -2:]
            pos = self.get_pos_emb(self.pos_emb, pos)
            state = torch.cat([color, shape, pos[:, :, 0], pos[:, :, 1]], dim=-1)
            state = self.obj_emb(state)

        if self.config.cw_embedding:
            B, K, _ = state.shape
            new_state = torch.zeros(B, K, self.rep_dim, device=state.device)
            arm_state = state[:, 0, :28]
            arm_state = self.get_pos_emb(self.cw_emb, arm_state)
            arm_state = self.arm_emb(arm_state.view(B, -1))
            new_state[:, 0] = arm_state

            obj_states = state[:, 1:, 28:]
            obj_pos = obj_states[:, :, :3]
            obj_pos = self.get_pos_emb(self.cw_emb, obj_pos.reshape(-1, 3)).reshape(B, K-1, -1)
            obj_colors = obj_states[:, :, 7:10]
            obj_states = self.obj_emb(torch.cat([obj_pos, obj_colors], dim=-1))
            new_state[:, 1:] = obj_states
            state = new_state

        if self.config.use_mlp1 or self.config.use_mlp2:
            state = self.mlp(state)


        return self._trans(state)
