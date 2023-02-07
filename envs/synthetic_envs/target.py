import numpy as np
from gym import spaces
from matplotlib import colors
from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite

from .base import BaseEnv
from .utils import norm


class TargetEnv(BaseEnv):
    def __init__(self, config, seed):
        super(TargetEnv, self).__init__(config, seed)
        self._target = config.target

    def _set_objs(self):
        objs = super()._set_objs()
        target_obj_idx = self._target_obj_idx = np.random.randint(self._num_objects)
        for n_idx in range(self._num_objects):
            if n_idx == target_obj_idx:
                objs[n_idx][0] = self._target[0]
                objs[n_idx][1] = self._target[1]
                objs[n_idx][2] = self._target[2]
            else:
                found = False
                while not found:
                    color = np.random.choice(self._COLORS)
                    shape = np.random.choice(self._SHAPES)
                    scale = np.random.choice(self._SCALES)
                    if (
                        self._target[0] == color
                        and self._target[1] == shape
                        and self._target[2] == scale
                    ):
                        found = False
                    else:
                        found = True
                objs[n_idx][0] = color
                objs[n_idx][1] = shape
                objs[n_idx][2] = scale
        objs = self._fill_positions(
            objs,
            agent_eps=self._config.distance_to_agent,
            objs_eps=self._config.distance_to_objs,
            wall_eps=self._config.distance_to_wall,
        )
        return objs

    def step(self, act):
        reward, is_success, done = super().step(act)
        reward, is_success, done = self._cal_reward(reward, is_success, done)
        return (
            self.render(),
            reward,
            done,
            {"is_success": is_success},
        )
