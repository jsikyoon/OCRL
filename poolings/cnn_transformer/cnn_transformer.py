from utils.tools import *
from poolings.base import Base
from .cnn_transformer_module import CNN_Transformer_Module


class CNN_Transformer(Base):
    def __init__(self, ocr, config: dict, num_stacked_obss: int=1) -> None:
        self._module = CNN_Transformer_Module(ocr.rep_dim, ocr.num_slots, config, num_stacked_obss)
        super(CNN_Transformer, self).__init__(ocr, config)
