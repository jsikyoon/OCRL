import torch
from torch import nn
from torch.nn import functional as F
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from utils.tools import *
from ocrs.base import Base
from .mae_module import MAE_Module


class MAE(Base):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        self._module = MAE_Module(ocr_config, env_config)
        super(MAE, self).__init__(ocr_config, env_config)
        param_groups = optim_factory.add_weight_decay(
                self._module,
                ocr_config.learning.weight_decay
        )
        self._opt = torch.optim.AdamW(
                self._module.parameters(),
                lr=ocr_config.learning.lr,
                betas=(0.9, 0.95)
        )
