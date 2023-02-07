import torch
from utils.tools import *


class Base:
    def __init__(self, ocr, config: dict) -> None:
        super().__init__()
        self._ocr = ocr
        self._config = config
        self._learn_aux_loss = config.learn_aux_loss
        self._learn_downstream_loss = config.learn_downstream_loss
        # load OCR
        self._load_ocr()

        # For downstream tasks
        self.rep_dim = self._module.rep_dim

        # optimizer
        if hasattr(self._config, "learning"):
            if hasattr(self._config.learning, "lr"):
                self._opt = torch.optim.Adam(
                    self._module.parameters(), lr=config.learning.lr
                )

    def _load_ocr(self):
        if self._config.ocr_checkpoint.run_id != "":
            checkpoint = torch.load(
                get_ocr_checkpoint_path(self._config.ocr_checkpoint), map_location="cpu"
            )
            self._ocr.load(checkpoint)

    def wandb_watch(self, config):
        wandb.watch(self._module, log=config.log)
        self._ocr.wandb_watch(config)

    def set_zero_grad(self):
        if hasattr(self, "_opt"):
            self._opt.zero_grad()
        self._ocr.set_zero_grad()

    def do_step(self):
        if hasattr(self, "_opt"):
            self._opt.step()
        self._ocr.do_step()

    def __call__(self, obs: Tensor, with_loss=False) -> Tensor:
        if self._learn_aux_loss and with_loss:
            metrics, state = self._ocr.get_loss(obs, with_rep=True)
            metrics["aux_loss"] = metrics["loss"]
            del metrics["loss"]
        else:
            state = self._ocr(obs)
            metrics = {}
        # detach if not fine-tuning
        state = state.detach() if not self._learn_downstream_loss else state
        # pooling
        state = self._module(state)
        return (state, metrics) if with_loss else state

    def train(self) -> None:
        self._module.train()
        self._ocr.train()
        return None

    def eval(self) -> None:
        self._module.eval()
        self._ocr.eval()
        return None

    def to(self, device: str) -> None:
        self._module.to(device)
        self._ocr.to(device)

    def get_samples(self, obs: Tensor) -> dict:
        return self._ocr.get_samples(obs)

    def save(self) -> dict:
        checkpoint = {}
        checkpoint["pooling_module_state_dict"] = self._module.state_dict()
        if hasattr(self, "_opt"):
            checkpoint["pooling_opt_state_dict"] = self._opt.state_dict()
        checkpoint.update(self._ocr.save())
        return checkpoint

    def load(self, checkpoint) -> None:
        self._module.load_state_dict(checkpoint["pooling_module_state_dict"])
        if hasattr(self, "_opt"):
            self._opt.load_state_dict(checkpoint["pooling_opt_state_dict"])
        self._ocr.load(checkpoint)
