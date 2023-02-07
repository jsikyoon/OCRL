import torch
from torch import nn
from torch.nn import functional as F

from utils.tools import *
from ocrs.base import Base
from .naturecnn_module import NatureCNN_Module


class NatureCNN(Base):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        self._module = NatureCNN_Module(ocr_config, env_config)
        super(NatureCNN, self).__init__(ocr_config, env_config)

    def get_samples(self, obs: Tensor) -> dict:
        return {}
