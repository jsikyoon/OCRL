from utils.tools import *
from poolings.base import Base
from .cnn_linear_module import CNN_Linear_Module


class CNN_Linear(Base):
    def __init__(self, ocr, config: dict, num_stacked_obss: int=1) -> None:
        self._module = CNN_Linear_Module(ocr.rep_dim, ocr.num_slots, config, num_stacked_obss)
        super(CNN_Linear, self).__init__(ocr, config)
