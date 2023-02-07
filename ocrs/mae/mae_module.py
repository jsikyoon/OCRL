# Taken from https://visionhong.tistory.com/38
import torch
from torch import nn
from torch.nn import functional as F

from utils.tools import *
from ocrs.base import Base
from .models_mae import mae_vit_base, mae_vit_large


class MAE_Module(nn.Module):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        super(MAE_Module, self).__init__()
        self._masking_ratio = ocr_config.masking_ratio
        self._return_cls = ocr_config.return_cls
        assert env_config.obs_size % ocr_config.patch_size == 0
        if ocr_config.vit_size == "base":
            self._mae = mae_vit_base(
                    img_size=env_config.obs_size,
                    patch_size=ocr_config.patch_size
            )
            self.rep_dim = 768
        elif ocr_config.vit_size == "large":
            self._mae = mae_vit_large(
                    img_size=env_config.obs_size,
                    patch_size=ocr_config.patch_size
            )
            self.rep_dim = 1024
        if self._return_cls:
            self.num_slots = 1
        else:
            self.num_slots = (env_config.obs_size // ocr_config.patch_size)**2
        self._patch_size = ocr_config.patch_size

    def forward(self, obs: Tensor) -> Tensor:
        # [B, num_patches+1, D]
        rep = self._mae.encode_full_patches(obs)
        if self._return_cls:
            return rep[:,0]
        else:
            return rep[:,1:]

    def get_loss(self, obs: Tensor, with_rep=False) -> Tensor:
        loss, pred, mask = self._mae(obs, self._masking_ratio)
        metrics = {"loss": loss, "mse": loss.detach()}
        if with_rep:
            return metrics, self(obs)
        else:
            return metrics

    def get_samples(self, obs: Tensor) -> dict:
        if self._masking_ratio == 0.0:
            _, pred, mask = self._mae(obs, 0.0)
            pred = self._mae.unpatchify(pred)
            return {"samples": np.concatenate([for_viz(obs), for_viz(pred)], axis=-2)}
        else:
            _, pred, mask = self._mae(obs, self._masking_ratio)
            mask = mask.unsqueeze(-1).repeat(1,1,self._patch_size**2 *3)
            mask = self._mae.unpatchify(mask)
            im_masked = obs * (1-mask)
            pred = self._mae.unpatchify(pred)
            im_paste = obs * (1-mask) + pred * mask
            return {"samples": np.concatenate([for_viz(obs), for_viz(im_masked), for_viz(im_paste)], axis=-2)}
