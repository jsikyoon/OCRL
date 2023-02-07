from utils.tools import *
from poolings.base import Base
from .identity_module import Identity_Module


class Identity(Base):
    def __init__(self, ocr, config: dict, num_stacked_obss: int=1) -> None:
        self._module = Identity_Module(ocr.rep_dim, ocr.num_slots, config, num_stacked_obss)
        super(Identity, self).__init__(ocr, config)
