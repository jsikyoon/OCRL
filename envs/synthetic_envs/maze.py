import copy
import numpy as np
from gym import spaces
from matplotlib import colors
from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite

from .base import BaseEnv
from .utils import norm


class MazeEnv(BaseEnv):
    def __init__(self, config, seed):
        super(MazeEnv, self).__init__(config, seed)
        self._walls = [
            # [0.25, [0.6, 1.0]],
            # [0.50, [0.0, 0.4]],
            # [0.75, [0.6, 1.0]],
            #[0.50, [0.6, 1.0]],
        ]
        # [agent_x, agent_y, goal_x, goal_y]
        self._task_types = [
                #[0.075, 0.075, 1.0, 1.0],
                #[0.075, 0.925, 1.0, 0.0],
                #[0.925, 0.075, 0.0, 1.0],
                #[0.925, 0.925, 0.0, 0.0],
                [None, None, 0.5, 0.5],
        ]
        self._goal = None
        if self._rew_type == "dense":
            self._dense_rews = [0.1] * len(self._walls)

        # Goal area must be included in the gt state
        if self.render_mode == "state":
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(self._num_objs_range[1] + 2, config.state_size),
                dtype=np.float,
            )

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
        task_type = self._task_types[np.random.randint(len(self._task_types))]
        self._goal = task_type[-2:]
        if task_type[0] is not None:
            objs[-1,3:5] = task_type[:2]
        goal = np.zeros((5), dtype=object) - 1
        goal[3:5] = task_type[-2:]
        objs = np.insert(objs, self._num_objects, goal, axis=0)
        return objs

    def _cal_reward(self, reward, is_success, done):
        if norm(self._goal - self._objs[-1, 3:5]) < self._AGENT[2]/2:
            reward = 1.0
            done = True
            is_success = True
        return reward, is_success, done

    def reset(self):
        self._dense_rews = [0.1] * len(self._walls)
        return super().reset()

    def _move_objs(self, idx, delta, eps=1 - 6):
        before_pos = copy.deepcopy(self._objs[-1, idx])
        self._objs[-1, idx] += delta
        self._objs[-1, idx] = np.clip(
            self._objs[-1, idx], self._AGENT[2] / 2, 1 - self._AGENT[2] / 2
        )
        for wall in self._walls:
            if (self._objs[-1, 3] - self._objs[-1, 2] / 2 < wall[0]) and (
                self._objs[-1, 3] + self._objs[-1, 2] / 2 > wall[0]
            ):
                if (self._objs[-1, 4] - self._objs[-1, 2] / 2 >= wall[1][0]) and (
                    self._objs[-1, 4] + self._objs[-1, 2] / 2 <= wall[1][1]
                ):
                    break
                else:
                    self._objs[-1, idx] = before_pos
                    break

    def step(self, act):
        """
        act: {0,1,2,3} <- up, left, down, right
        """
        reward, is_success, done = 0.0, False, False
        # move
        if act == 0:
            self._move_objs(4, self._moving_step_size)
        elif act == 1:
            self._move_objs(3, -1 * self._moving_step_size)
        elif act == 2:
            self._move_objs(4, -1 * self._moving_step_size)
        elif act == 3:
            self._move_objs(3, self._moving_step_size)
        else:
            raise ValueError(f"action must be one of {0,1,2,3}, not {act}")
        self.step_count += 1
        if self.step_count >= self._max_steps:
            done = True
        # dense reward type
        if self._rew_type == "dense":
            for w_idx in range(len(self._walls)):
                if w_idx == 0:
                    low = 0.0
                else:
                    low = self._walls[w_idx-1][0]
                high = self._walls[w_idx][0]
                if (self._objs[-1][3] >= low) and (self._objs[-1][3] <= high):
                    reward = self._dense_rews[w_idx]
                    self._dense_rews[w_idx] = 0.0
                    break
        reward, is_success, done = self._cal_reward(reward, is_success, done)
        return (
            self.render(),
            reward,
            done,
            {"is_success": is_success},
        )
