import gym
import wandb
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from utils.tools import *
import ocrs
import poolings


class OCRExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param rep_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, config=None):
        ocr = getattr(ocrs, config.ocr.name)(config.ocr, config.env)
        pooling = getattr(poolings, config.pooling.name)(ocr, config.pooling)
        super(OCRExtractor, self).__init__(observation_space, pooling.rep_dim)
        del ocr, pooling
        self._num_stacked_obss = config.env.num_stacked_obss
        self._obs_size = config.env.obs_size
        self._obs_channels = config.env.obs_channels
        self._num_envs = config.num_envs
        self._viz_step = 0
        self._viz_interval = config.viz_interval
        self._ocr, self._ocr_pretraining = get_ocr(
            config.ocr, config.env, config.pooling.ocr_checkpoint, config.device
        )
        self._pooling = getattr(poolings, config.pooling.name + "_Module")(
            self._ocr.rep_dim, self._ocr.num_slots, config.pooling
        )

    def forward(self, observations: Tensor) -> Tensor:
        if self._ocr_pretraining:
            if observations.shape[0] == self._num_envs:
                if self._viz_step % self._viz_interval == 0:
                    samples = self._ocr.get_samples(observations)
                    wandb.log(
                        {k: [wandb.Image(_v) for _v in v] for k, v in samples.items()},
                    )
                self._viz_step += 1
        return self._pooling(self._ocr(observations))
