import wandb
import torch
from torch.nn.utils import clip_grad_norm_

from utils.tools import *


class Base:
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        super().__init__()
        self.name = ocr_config.name
        self._config = ocr_config
        self._obs_size = env_config.obs_size
        self._obs_channels = env_config.obs_channels

        # for pooling layer
        self.rep_dim = self._module.rep_dim
        self.num_slots = self._module.num_slots

        # optimizer
        if hasattr(self._config, "learning"):
            if hasattr(self._config.learning, "lr"):
                self._opt = torch.optim.Adam(
                    self._module.parameters(), lr=self._config.learning.lr
                )

    def __call__(self, obs: Tensor) -> Tensor:
        return self._module(obs)

    def wandb_watch(self, config):
        wandb.watch(self._module, log=config.log)

    def get_loss(self, obs: Tensor, with_rep=False) -> dict:
        return self._module.get_loss(obs, with_rep)

    def train(self) -> None:
        self._module.train()
        return None

    def eval(self) -> None:
        self._module.eval()
        return None

    def to(self, device: str) -> None:
        self._module.to(device)
        if hasattr(self, "_opt"):
            optimizer_to(self._opt, device)

    def set_zero_grad(self):
        if hasattr(self, "_opt"):
            self._opt.zero_grad()

    def do_step(self):
        if hasattr(self, "_opt"):
            self._opt.step()

    def get_samples(self, obs: Tensor) -> dict:
        return self._module.get_samples(obs)

    def update(self, obs: Tensor, masks: Tensor, step: int) -> dict:
        if hasattr(self, "_opt"):
            self._opt.zero_grad()
            metrics = self.get_loss(obs, masks)
            metrics["loss"].backward()
            if hasattr(self._config.learning, "clip"):
                clip_norm_type = self._config.learning.clip_norm_type if hasattr(self._config.learning, "clip_norm_type") else "inf"
                norm = clip_grad_norm_(
                    self._module.parameters(), self._config.learning.clip, clip_norm_type
                )
                metrics["norm"] = norm
            self._opt.step()
            return metrics
        else:
            return {}

    def save(self) -> dict:
        checkpoint = {}
        checkpoint["ocr_module_state_dict"] = self._module.state_dict()
        if hasattr(self, "_opt"):
            checkpoint["ocr_opt_state_dict"] = self._opt.state_dict()
        return checkpoint

    def load(self, checkpoint: str) -> None:
        self._module.load_state_dict(checkpoint["ocr_module_state_dict"])
        # for name1, name2 in zip(self._module.state_dict().keys(), checkpoint["ocr_nets_state_dict"].keys()):
        #    self._module.state_dict()[name1] = checkpoint["ocr_nets_state_dict"][name2]
        if hasattr(self, "_opt"):
            self._opt.load_state_dict(checkpoint["ocr_opt_state_dict"])
