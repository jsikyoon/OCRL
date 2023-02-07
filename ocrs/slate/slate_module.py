import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from itertools import chain

from utils.tools import *
from ocrs.common.utils import (
    cosine_anneal,
    linear_warmup,
    gumbel_softmax,
    PositionalEmbedding,
)
from ocrs.common.networks import linear
from ocrs.common.models import SlotAttnCNNEncoder, dVAE
from ocrs.common.slot_attn import SlotAttentionEncoder
from ocrs.common.transformer import LearnedPositionalEncoding, TransformerDecoder

from ocrs.common.models import BroadCastDecoder


class SLATE_Module(nn.Module):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        super(SLATE_Module, self).__init__()
        self._obs_size = obs_size = env_config.obs_size
        self._obs_channels = obs_channels = env_config.obs_channels
        self._use_cnn_feat = use_cnn_feat = ocr_config.use_cnn_feat
        self._use_bcdec = ocr_config.use_bcdec

        ## Configs
        # dvae config
        self._vocab_size = vocab_size = ocr_config.dvae.vocab_size
        self._d_model = d_model = ocr_config.dvae.d_model
        # SA config
        cnn_hsize = ocr_config.cnn.hidden_size
        num_iterations = ocr_config.slotattr.num_iterations
        self._num_slots = num_slots = ocr_config.slotattr.num_slots
        slot_size = ocr_config.slotattr.slot_size
        mlp_hidden_size = ocr_config.slotattr.mlp_hidden_size
        pos_channels = ocr_config.slotattr.pos_channels
        num_slot_heads = ocr_config.slotattr.num_slot_heads
        # tfdec config
        num_dec_blocks = ocr_config.tfdec.num_dec_blocks
        num_dec_heads = ocr_config.tfdec.num_dec_heads
        # learning config
        dropout = ocr_config.learning.dropout
        self._tau_start = ocr_config.tau_start
        self._tau_final = ocr_config.tau_final
        self._tau_steps = ocr_config.tau_steps
        self._tau = 1.0

        self._hard = ocr_config.hard

        # build Discrete VAE
        self._dvae = dVAE(self._vocab_size, obs_channels)
        self._enc_size = enc_size = obs_size // 4
        # build encoder
        self._enc = SlotAttnCNNEncoder(obs_size, obs_channels, cnn_hsize)
        self._enc_pos = PositionalEmbedding(obs_size, cnn_hsize)
        self._slotattn = SlotAttentionEncoder(
            num_iterations,
            num_slots,
            cnn_hsize,
            slot_size,
            mlp_hidden_size,
            pos_channels,
            num_slot_heads,
        )

        # broadcast decoder
        if self._use_bcdec:
            self._dec = BroadCastDecoder(obs_size, obs_channels, cnn_hsize, slot_size)

        self._slotproj = linear(slot_size, d_model, bias=False)
        # build decoder
        self._dict = OneHotDictionary(vocab_size, d_model)
        self._bos_token = BosToken(d_model)
        self._z_pos = LearnedPositionalEncoding(1 + enc_size**2, d_model, dropout)
        self._tfdec = TransformerDecoder(
            num_dec_blocks, enc_size**2, d_model, num_dec_heads, dropout
        )
        # intermediate layer between TF decoder and dVAE decoder
        self._out = linear(d_model, vocab_size, bias=False)

        # For pooling layer
        if use_cnn_feat:
            self.num_slots = obs_size**2
            self.rep_dim = cnn_hsize + obs_channels
        else:
            self.num_slots = num_slots
            self.rep_dim = slot_size

    def get_dvae_params(self):
        return self._dvae.parameters()

    def get_sa_params(self):
        if self._use_bcdec:
            return chain(
                self._enc.parameters(),
                self._enc_pos.parameters(),
                self._slotattn.parameters(),
                self._slotproj.parameters(),
                self._dec.parameters(),
            )
        else:
            return chain(
                self._enc.parameters(),
                self._enc_pos.parameters(),
                self._slotattn.parameters(),
                self._slotproj.parameters(),
            )

    def get_tfdec_params(self):
        return chain(
            self._dict.parameters(),
            self._bos_token.parameters(),
            self._z_pos.parameters(),
            self._tfdec.parameters(),
            self._out.parameters(),
        )

    def _get_z(self, obs):
        # dvae encode
        z, z_logits = self._dvae(obs, self._tau, self._hard)
        # hard z
        z_hard = gumbel_softmax(z_logits, self._tau, True, dim=1).detach()
        return z, z_hard

    def _get_slots(self, obs, with_attns=False, z_hard=None, with_ce=False):
        res = []
        emb = self._enc_pos(self._enc(obs))
        emb = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        # apply slot attention
        slots, attns = self._slotattn(emb)
        res.append(slots)
        if with_attns:
            res.append(attns)
        if with_ce:
            # target tokens for transformer
            z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
            z_emb = self._dict(z_hard)
            z_emb = torch.cat(
                [self._bos_token().expand(obs.shape[0], -1, -1), z_emb], dim=1
            )
            z_emb = self._z_pos(z_emb)
            # to calculate cross entropy
            projected_slots = self._slotproj(slots)
            decoder_output = self._tfdec(z_emb[:, :-1], projected_slots)
            pred = self._out(decoder_output)
            cross_entropy = (
                -(z_hard * torch.log_softmax(pred, dim=-1))
                .flatten(start_dim=1)
                .sum(-1)
                .mean()
            )
            res.append(cross_entropy)
        if len(res) == 1:
            return res[0]
        else:
            return tuple(res)

    def _gen_imgs(self, slots):
        slots = self._slotproj(slots)
        z_gen = slots.new_zeros(0)
        _input = self._bos_token().expand(slots.shape[0], 1, -1)
        for t in range(self._enc_size**2):
            decoder_output = self._tfdec(self._z_pos(_input), slots)
            z_next = F.one_hot(
                self._out(decoder_output)[:, -1:].argmax(dim=-1), self._vocab_size
            )
            z_gen = torch.cat((z_gen, z_next), dim=1)
            _input = torch.cat([_input, self._dict(z_next)], dim=1)
        z_gen = (
            z_gen.transpose(1, 2)
            .float()
            .reshape(slots.shape[0], -1, self._enc_size, self._enc_size)
        )
        return self._dvae.decode(z_gen)

    def forward(self, obs: Tensor, with_attns, with_masks) -> Tensor:
        assert not (with_attns and with_masks) # one of attns and masks can be returned
        if self._use_cnn_feat:
            return img_to_slot(torch.cat([self._enc_pos(self._enc(obs)), obs], dim=1))
            # return img_to_slot(self._enc_pos(self._enc(obs)))
        else:
            if with_attns or with_masks:
                slots, attns = self._get_slots(obs, with_attns=True)
                attns = attns.transpose(-1, -2).reshape(
                    obs.shape[0], slots.shape[1], 1, obs.shape[2], obs.shape[3]
                )
                if with_attns:
                    attns = obs.unsqueeze(1) * attns + (1.0 - attns)
                return slots, attns
            else:
                return self._get_slots(obs)

    def get_loss(self, obs: Tensor, masks: Tensor, with_rep=False, with_mse=False) -> dict:
        # get z
        z, z_hard = self._get_z(obs)
        B, _, H_enc, W_enc = z.size()
        # dvae recon
        recon = self._dvae.decode(z)
        dvae_mse = ((obs - recon) ** 2).sum() / obs.shape[0]
        # get slots
        slots, attns, cross_entropy = self._get_slots(obs, z_hard=z_hard, with_attns=True, with_ce=True)
        attns = attns.transpose(-1, -2).reshape(
            obs.shape[0], slots.shape[1], 1, obs.shape[2], obs.shape[3]
        )
        fg_mask = (1 - masks[:,-1].unsqueeze(1))
        attns = attns * fg_mask
        attns = torch.cat([attns, fg_mask], dim=1)
        ari = np.mean(calculate_ari(masks, attns))

        if self._use_bcdec:
            recon = self._dec(slots)
            mse = ((obs - recon) ** 2).sum() / obs.shape[0]
            metrics = {
                "loss": mse,
                "mse": mse.detach(),
                "ari": ari,
            }
        else:
            metrics = {
                "loss": dvae_mse + cross_entropy,
                "dvae_mse": dvae_mse.detach(),
                "ari": ari,
                "cross_entropy": cross_entropy.detach(),
                "tau": torch.Tensor([self._tau]),
            }
            # generate image tokens auto-regressively
            if with_mse:
                recon_tf = self._gen_imgs(slots)
                mse = ((obs - recon_tf) ** 2).sum() / obs.shape[0]
                metrics.update({"mse": mse.detach()})

        return (metrics, zs) if with_rep else metrics

    def get_samples(self, obs: Tensor) -> dict:
        # get z
        z, z_hard = self._get_z(obs)
        B, _, H_enc, W_enc = z.size()
        # dvae recon
        recon = self._dvae.decode(z)
        # get slots
        slots, attns = self._get_slots(obs, with_attns=True)
        attns = attns.transpose(-1, -2).reshape(
            obs.shape[0], self._num_slots, 1, self._obs_size, self._obs_size
        )
        attns = obs.unsqueeze(1) * attns + (1.0 - attns)
        if self._use_bcdec:
            recon = self._dec(slots)
            return {"samples": for_viz(visualize([obs, recon, attns]))}
        else:
            # generate image tokens auto-regressively
            recon_tf = self._gen_imgs(slots)
            return {"samples": for_viz(visualize([obs, recon, recon_tf, attns]))}

    def update_tau(self, step: int) -> None:
        # update tau
        self._tau = cosine_anneal(
            step, self._tau_start, self._tau_final, 0, self._tau_steps
        )


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """
        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs


class BosToken(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self._bos_token = nn.Parameter(torch.Tensor(1, 1, d_model))
        nn.init.xavier_uniform_(self._bos_token)

    def forward(self):
        return self._bos_token
