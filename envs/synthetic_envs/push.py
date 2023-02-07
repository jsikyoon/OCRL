import copy
import numpy as np
from gym import spaces
from matplotlib import colors
from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite

from .base import BaseEnv
from .utils import norm


class PushEnv(BaseEnv):
    def __init__(self, config, seed):
        super(PushEnv, self).__init__(config, seed)
        self._target = config.target

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
        #target_obj_idx = self._target_obj_idx = np.random.randint(self._num_objects)
        target_obj_idx = self._target_obj_idx = 0
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
        goal = np.zeros((5), dtype=object)
        goal[:3] = objs[target_obj_idx, :3]
        goal[3:5] = self._get_goal_position(objs[target_obj_idx, 2] / 2)
        objs = np.insert(objs, self._num_objects, goal, axis=0)
        objs = self._fill_positions(
            objs,
            agent_eps=self._config.distance_to_agent,
            objs_eps=self._config.distance_to_objs,
            wall_eps=self._config.distance_to_wall,
        )
        return objs

    def _get_goal_position(self, radius):
        #goal_positions = [[0.5, 0.5]]
        goal_positions = [[radius, radius]]
        return goal_positions[np.random.randint(len(goal_positions))]

    def _cal_reward(self, reward, is_success, done, eps=1e-6):
        for i in range(self._num_objects):
            if (
                norm(self._objs[i, 3:5] - self._objs[-2, 3:5]) + eps
                < self._objs[i, 2] / 2 + self._objs[-2, 2] / 2
            ):
                if i == self._target_obj_idx:
                    reward = 1.0
                    is_success = True
                else:
                    reward = 0.1 if self._rew_type == "normal" else 0.0
                    is_success = False
                done = True
                break
        return reward, is_success, done

    def _check_can_move(self, obj_idx, idx, eps=1e-6):
        for i in range(self._num_objects):
            if i == obj_idx:
                continue
            if (
                norm(self._objs[i, 3:5] - self._objs[obj_idx, 3:5]) + eps
                < self._objs[i, 2] / 2 + self._objs[obj_idx, 2] / 2
            ):
                return False
        return True

    def _move_objs(self, idx, delta, eps=1e-6):
        self._objs[-1, idx] += delta
        moves = [delta]
        moved_objs = []
        objs_before_move = copy.deepcopy(self._objs)
        for i in range(self._num_objects):
            if (
                norm(self._objs[i, 3:5] - self._objs[-1, 3:5]) + eps
                < self._objs[i, 2] / 2 + self._AGENT[2] / 2
            ):
                # when object is near the wall
                if (self._objs[i, idx] == self._objs[i, 2] / 2) or (
                    self._objs[i, idx] == 1 - self._objs[i, 2] / 2
                ):
                    moves.append(0)
                    break
                before_pos = copy.deepcopy(self._objs[i, idx])
                self._objs[i, idx] += delta
                if not self._check_can_move(i, idx):
                    self._objs[i, idx] -= delta
                    moves.append(0)
                    break
                self._objs[i, idx] = np.clip(
                    self._objs[i, idx], self._objs[i, 2] / 2, 1 - self._objs[i, 2] / 2
                )
                moves.append(self._objs[i, idx] - before_pos)
                moved_objs.append(i)
        #if len(moved_objs) > 1:
        #    for i in moved_objs:
        #        self._objs[i, idx] = objs_before_move[i, idx]
        #    moves.append(0)
        if delta > 0:
            self._objs[-1, idx] = self._objs[-1, idx] - delta + np.min(moves)
        else:
            self._objs[-1, idx] = self._objs[-1, idx] - delta + np.max(moves)

    def step(self, act):
        """
        act: {0,1,2,3} <- up, left, down, right
        """
        reward, is_success, done = 0.0, False, False
        dist_before_acting = self._get_dist(self._target_obj_idx, -2)
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
        self._objs[-1, 3] = np.clip(
            self._objs[-1, 3], self._AGENT[2] / 2, 1 - self._AGENT[2] / 2
        )
        self._objs[-1, 4] = np.clip(
            self._objs[-1, 4], self._AGENT[2] / 2, 1 - self._AGENT[2] / 2
        )
        self.step_count += 1
        if self.step_count >= self._max_steps:
            done = True
        # dense reward type
        if self._rew_type == "dense":
            # additional reward like object-centric intrinsic rewards
            if self._get_dist(self._target_obj_idx, -2) != dist_before_acting:
                reward = 0.01
            else:
                reward = 0.0
        reward, is_success, done = self._cal_reward(reward, is_success, done)
        return (
            self.render(),
            reward,
            done,
            {"is_success": is_success},
        )
