from utils.tools import *
from poolings.base import Base
from .rn_module import RN_Module


class RN(Base):
    def __init__(self, ocr, num_stacked_obss: int, config: dict) -> None:
        self._module = RN_Module(ocr.rep_dim, ocr.num_slots, num_stacked_obss, config)
        super(RN, self).__init__(ocr, config)
