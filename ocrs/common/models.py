import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ocrs.common.networks import Conv2dBlock, conv2d
from ocrs.common.utils import PositionalEmbedding, gumbel_softmax


class dVAE(nn.Module):
    def __init__(self, vocab_size, img_channels):
        super().__init__()

        self._encoder = nn.Sequential(
            Conv2dBlock(img_channels, 64, 4, 4),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            conv2d(64, vocab_size, 1),
        )
        self._decoder = nn.Sequential(
            Conv2dBlock(vocab_size, 64, 1),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            conv2d(64, img_channels, 1),
        )

    def forward(self, obs, tau=1.0, hard=True):
        z_logits = F.log_softmax(self._encoder(obs), dim=1)
        z = gumbel_softmax(z_logits, tau, hard, dim=1)
        return z, z_logits

    def decode(self, z):
        return self._decoder(z)


# Used for (D)VAE encoder
class VAEEncoder(nn.Module):
    def __init__(self, obs_channels, out_dim, compression_rate):
        super().__init__()
        self._encoder = []
        in_channels = obs_channels
        assert np.log2(compression_rate).is_integer()
        for i in range(int(np.log2(compression_rate))):
            self._encoder.extend(
                [
                    Conv2dBlock(in_channels, 64, 2, 2),
                    Conv2dBlock(64, 64, 1, 1),
                    Conv2dBlock(64, 64, 1, 1),
                    Conv2dBlock(64, 64, 1, 1),
                ]
            )
            in_channels = 64
        self._encoder.append(conv2d(64, out_dim, 1))
        self._encoder = nn.Sequential(*self._encoder)

    def forward(self, obs):
        return self._encoder(obs)


# Used for (D)VAE decoder
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, obs_channels, compression_rate):
        super().__init__()
        self._decoder = []
        self._decoder.append(Conv2dBlock(latent_dim, 64, 1))
        assert np.log2(compression_rate).is_integer()
        for i in range(int(np.log2(compression_rate))):
            self._decoder.extend(
                [
                    Conv2dBlock(64, 64, 3, 1, 1),
                    Conv2dBlock(64, 64, 1, 1),
                    Conv2dBlock(64, 64, 1, 1),
                    Conv2dBlock(64, 64 * 2 * 2, 1),
                    nn.PixelShuffle(2),
                ]
            )
        self._decoder.append(conv2d(64, obs_channels, 1))
        self._decoder = nn.Sequential(*self._decoder)

    def forward(self, obs):
        return self._decoder(obs)


class SlotAttnCNNEncoder(nn.Module):
    def __init__(self, obs_size, obs_channels, hidden_size):
        super().__init__()
        self._encoder = nn.Sequential(
            Conv2dBlock(obs_channels, hidden_size, 5, 1, 2),
            Conv2dBlock(hidden_size, hidden_size, 5, 1, 2),
            Conv2dBlock(hidden_size, hidden_size, 5, 1, 2),
            conv2d(hidden_size, hidden_size, 5, 1, 2),
        )

    def forward(self, obs):
        return self._encoder(obs)


class BroadCastDecoder(nn.Module):
    def __init__(self, obs_size, obs_channels, hidden_size, slot_size):
        super().__init__()
        self._obs_size = obs_size
        self._obs_channels = obs_channels
        self._decoder = nn.Sequential(
            Conv2dBlock(slot_size, hidden_size, 5, 1, 2),
            Conv2dBlock(hidden_size, hidden_size, 5, 1, 2),
            Conv2dBlock(hidden_size, hidden_size, 5, 1, 2),
            conv2d(hidden_size, obs_channels + 1, 3, 1, 1),
        )
        self._pos_emb = PositionalEmbedding(obs_size, slot_size)

    def _spatial_broadcast(self, slots):
        slots = slots.unsqueeze(-1).unsqueeze(-1)
        return slots.repeat(1, 1, self._obs_size, self._obs_size)

    def forward(self, slots):
        B, N, _ = slots.shape
        # [batch_size * num_slots, d_slots]
        slots = slots.flatten(0, 1)
        # [batch_size * num_slots, d_slots, obs_size, obs_size]
        slots = self._spatial_broadcast(slots)
        out = self._decoder(self._pos_emb(slots))
        img_slots, masks = out[:, : self._obs_channels], out[:, -1:]
        img_slots = img_slots.view(
            B, N, self._obs_channels, self._obs_size, self._obs_size
        )
        masks = masks.view(B, N, 1, self._obs_size, self._obs_size)
        masks = masks.softmax(dim=1)
        recon_slots_masked = img_slots * masks
        return recon_slots_masked.sum(dim=1)
