import numpy as np
from gym import spaces
from matplotlib import colors
from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite

from .base import BaseEnv
from .utils import norm


class RandomObjsEnv(BaseEnv):
    def __init__(self, config, seed):
        super(RandomObjsEnv, self).__init__(config, seed)

    def _set_objs(self):
        objs = super()._set_objs()
        for n_idx in range(self._num_objects):
            objs[n_idx][0] = np.random.choice(self._COLORS)
            objs[n_idx][1] = np.random.choice(self._SHAPES)
            objs[n_idx][2] = np.random.choice(self._SCALES)
        objs = self._fill_positions(
            objs,
            agent_eps=self._config.distance_to_agent,
            objs_eps=self._config.distance_to_objs,
            wall_eps=self._config.distance_to_wall,
        )
        return objs

    def _cal_reward(self, reward, is_success, done):
        for i in range(self._num_objects):
            if norm(self._objs[i, 3:5] - self._objs[-1, 3:5]) < self._AGENT[2]:
                reward = 1.0
                done = True
                is_success = True
                break
        return reward, is_success, done

    def step(self, act):
        reward, is_success, done = super().step(act)
        reward, is_success, done = self._cal_reward(reward, is_success, done)
        return (
            self.render(),
            reward,
            done,
            {"is_success": is_success},
        )
