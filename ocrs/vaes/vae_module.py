import torch
from torch import nn
from torch.nn import functional as F

from utils.tools import *
from ocrs.common.models import VAEEncoder, VAEDecoder
from ocrs.base import Base


class VAE_Module(nn.Module):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        super(VAE_Module, self).__init__()
        obs_size = env_config.obs_size
        obs_channels = env_config.obs_channels
        latent_dim = ocr_config.latent_dim
        self._kld_weight = ocr_config.learning.kld_weight
        self._use_cnn_feat = ocr_config.use_cnn_feat
        self._cnn_feat_size = cnn_feat_size = ocr_config.cnn_feat_size

        if self._use_cnn_feat:
            self.rep_dim = 64
            self.num_slots = cnn_feat_size**2
        else:
            self.rep_dim = latent_dim
            self.num_slots = 1

        # build encoder
        self._enc = VAEEncoder(obs_channels, 64, obs_size // cnn_feat_size)
        # mu and variance layers
        self._mu = nn.Linear(64 * cnn_feat_size * cnn_feat_size, latent_dim)
        self._var = nn.Linear(64 * cnn_feat_size * cnn_feat_size, latent_dim)
        # input layer of decoder
        self._in_dec = nn.Linear(latent_dim, 64 * cnn_feat_size * cnn_feat_size)
        # build decoder
        self._dec = VAEDecoder(64, obs_channels, obs_size // cnn_feat_size)

    def _encode(self, obs: Tensor) -> List[Tensor]:
        out = self._enc(obs).reshape(obs.shape[0], -1)
        return self._get_mu_logvar(out)

    def _get_mu_logvar(self, out: Tensor) -> List[Tensor]:
        mu = self._mu(out)
        log_var = self._var(out)
        return mu, log_var

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, obs: Tensor) -> Tensor:
        return (
            img_to_slot(self._enc(obs)) if self._use_cnn_feat else self._encode(obs)[0]
        )

    def get_loss(self, obs: Tensor, with_rep=False) -> Tensor:
        if self._use_cnn_feat:
            rep = self._enc(obs)
            mu, log_var = self._get_mu_logvar(rep.reshape(rep.shape[0], -1))
            rep = img_to_slot(rep)
        else:
            mu, log_var = self._encode(obs)
            rep = mu
        latent = self._reparameterize(mu, log_var)
        in_dec = self._in_dec(latent).reshape(
            obs.shape[0], 64, self._cnn_feat_size, self._cnn_feat_size
        )
        recon = self._dec(in_dec)
        mse = ((obs - recon) ** 2).sum() / obs.shape[0]
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        loss = mse + self._kld_weight * kld_loss
        metrics = {"loss": loss, "mse": mse.detach(), "kld": -kld_loss.detach()}
        if with_rep:
            return metrics, rep
        else:
            return metrics

    def get_samples(self, obs: Tensor) -> dict:
        mu, log_var = self._encode(obs)
        latent = self._reparameterize(mu, log_var)
        latent = self._in_dec(latent).reshape(
            obs.shape[0], 64, self._cnn_feat_size, self._cnn_feat_size
        )
        recon = self._dec(latent)
        return {"samples": np.concatenate([for_viz(obs), for_viz(recon)], axis=-2)}
