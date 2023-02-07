from ocrs.base import Base
from .iodine_module import Iodine_Module

class Iodine(Base):
    def __init__(self, ocr_config: dict, env_config: dict) -> None:
        self._module = Iodine_Module(ocr_config, env_config)
        super(Iodine, self).__init__(ocr_config, env_config)

    def __call__(self, obs, with_masks=False):
        return self._module(obs, with_masks)
    
    def get_loss(self, obs, masks, with_rep=False) -> dict:
        metrics = self._module.get_loss(obs, masks, False)
        return metrics
