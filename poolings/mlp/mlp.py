from utils.tools import *
from poolings.base import Base
from .mlp_module import MLP_Module


class MLP(Base):
    def __init__(self, ocr, config: dict, num_stacked_obss: int=1) -> None:
        self._module = MLP_Module(ocr.rep_dim, ocr.num_slots, config, num_stacked_obss)
        super(MLP, self).__init__(ocr, config)
