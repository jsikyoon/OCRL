import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler

from utils.tools import *
from ocrs.base import Base
from ocrs.common.utils import linear_warmup
from .slate_module import SLATE_Module


class SLATE(Base):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        self._module = SLATE_Module(ocr_config, env_config)
        super(SLATE, self).__init__(ocr_config, env_config)

        # optimizer
        self._opt = torch.optim.Adam(
            [
                {
                    "params": self._module.get_dvae_params(),
                    "lr": self._config.learning.lr_dvae,
                },
                {
                    "params": self._module.get_sa_params(),
                    "lr": self._config.learning.lr_enc,
                },
                {
                    "params": self._module.get_tfdec_params(),
                    "lr": self._config.learning.lr_dec,
                },
            ]
        )

    def __call__(self, obs: Tensor, with_attns=False, with_masks=False) -> Tensor:
        return self._module(obs, with_attns, with_masks)

    def get_loss(self, obs: Tensor, masks:Tensor, with_rep=False, with_mse=False) -> dict:
        if with_rep:
            metrics, slots = self._module.get_loss(obs, masks, with_rep, with_mse)
        else:
            metrics = self._module.get_loss(obs, masks, with_rep, with_mse)
        metrics.update(
            {
                "lr_dvae": torch.Tensor([self._opt.param_groups[0]["lr"]]),
                "lr_enc": torch.Tensor([self._opt.param_groups[1]["lr"]]),
                "lr_dec": torch.Tensor([self._opt.param_groups[2]["lr"]]),
            }
        )
        return (metrics, slots) if with_rep else metrics

    def update(self, obs: Tensor, masks: Tensor, step: int) -> dict:
        self._module.update_tau(step)
        # update lr
        lr_warmup_factor = linear_warmup(
            step, 0, 1, 0, self._config.learning.lr_warmup_steps
        )
        lr_decay_factor = math.exp(
            step / self._config.learning.lr_half_life * math.log(0.5)
        )
        lr_dvae = self._config.learning.lr_dvae
        lr_enc = lr_decay_factor * lr_warmup_factor * self._config.learning.lr_enc
        lr_dec = lr_decay_factor * lr_warmup_factor * self._config.learning.lr_dec
        self._opt.param_groups[0]["lr"] = lr_dvae
        self._opt.param_groups[1]["lr"] = lr_enc
        self._opt.param_groups[2]["lr"] = lr_dec
        # update params
        return super().update(obs, masks, step)
