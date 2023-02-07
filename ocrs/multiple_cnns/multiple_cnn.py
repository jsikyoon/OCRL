import torch
from torch import nn
from torch.nn import functional as F

from utils.tools import *
from ocrs.base import Base
from .multiple_cnn_module import MultipleCNN_Module


class MultipleCNN(Base):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        self._module = MultipleCNN_Module(ocr_config, env_config)
        super(MultipleCNN, self).__init__(ocr_config, env_config)

    def get_samples(self, obs: Tensor) -> dict:
        return {}
