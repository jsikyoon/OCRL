from utils.tools import *
from poolings.base import Base
from .transformer_module import Transformer_Module


class Transformer(Base):
    def __init__(self, ocr, config: dict, num_stacked_obss: int=1) -> None:
        self._module = Transformer_Module(ocr.rep_dim, ocr.num_slots, config, num_stacked_obss)
        super(Transformer, self).__init__(ocr, config)
