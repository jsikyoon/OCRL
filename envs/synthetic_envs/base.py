import os
import numpy as np
from PIL import Image
from gym import spaces
from pathlib import Path
from matplotlib import colors
from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite

from .utils import norm

COLORS = ["blue", "green", "yellow", "red", "cyan", "pink", "brown"]
SHAPES = ["square", "triangle", "star_4", "circle", "pentagon", "hexagon", "octagon", "star_5", "star_6", "spoke_4", "spoke_5", "spoke_6"]
SCALES = [0.15, 0.22]

class BaseEnv:
    metadata = {"render.modes": ["rgb_array", "state", "image", "mask"]}

    def __init__(self, config, seed):
        np.random.seed(seed)
        assert config.mode in ["easy", "normal", "hard"]
        assert config.rew_type in ["sparse", "normal", "dense"]
        self._name = config.name
        self._config = config
        self._mode = config.mode
        self._rew_type = config.rew_type
        self.render_mode = config.render_mode
        self._num_stacked_obss = config.num_stacked_obss
        self._obs_size = config.obs_size
        self._obs_channels = config.obs_channels
        self._num_objs_range = config.num_objects_range
        self._renderer = spriteworld_renderers.PILRenderer(
            image_size=(config.obs_size, config.obs_size),
            anti_aliasing=10,
        )
        self._moving_step_size = config.moving_step_size
        self._wo_agent = config.wo_agent
        self._skewed = config.skewed
        self._occlusion = config.occlusion
        self._max_steps = config.max_steps
        self._agent_pos = config.agent_pos
        self._COLORS = config.COLORS
        self._SHAPES = config.SHAPES
        self._SCALES = config.SCALES
        self._AGENT = config.AGENT

        # background
        self._use_bg = config.background.use_bg
        if self._use_bg:
            self._bg_imgs = []
            parent_dir = Path(__file__).resolve().parent.parent.parent
            for img_path in config.background.img_paths:
                img_list = os.listdir(os.path.join(parent_dir, img_path))
                for _img_name in img_list:
                    self._bg_imgs.append(os.path.join(parent_dir, img_path, _img_name))
            self._bg_imgs.append("Black")

        self.action_space = spaces.Discrete(4)
        if self.render_mode == "state":
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(self._num_objs_range[1] + 1, config.state_size),
                dtype=np.float,
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(
                    self._obs_size,
                    self._obs_size,
                    self._obs_channels * self._num_stacked_obss,
                ),
                dtype=np.uint8,
            )

        self._objs = None
        self.step_count = 0

    def _get_position(self, pos_min, pos_max, radius, eps):
        if pos_min == pos_max:
            return pos_min
        # when range is very small
        #if pos_min + radius + eps > pos_max - radius - eps:
        #    return np.random.uniform(pos_min, pos_max)
        if self._mode == "easy":
            return np.random.uniform(pos_min, pos_max)
        else:
            # when using specific range
            #if pos_min != 0.0:
            #    _min = pos_min
            #else:
            _min = pos_min + radius + eps
            # when using specific range
            #if pos_max != 1.0:
            #    _max = pos_max
            #else:
            _max = pos_max - radius - eps
            return np.random.uniform(_min, _max)

    def _fill_positions(
        self,
        objs,
        agent_eps=0.08,
        objs_eps=0.08,
        wall_eps=0.08,
        skew_mu=0.25,
        skew_sigma=0.1,
        occlusion_threshold=0.15,
    ):
        if self._agent_pos is not None:
            objs[-1, 3] = float(self._agent_pos[0])
            objs[-1, 4] = float(self._agent_pos[1])
        for i, _obj in enumerate(objs):
            # if the task is push, then the goal position is not randomly positioned
            if (i == len(objs)-2) and ("Push" in self._name):
                continue
            # agent
            if (i == len(objs)-1) and (self._agent_pos is not None or self._wo_agent):
                continue
            x_min, x_max, y_min, y_max = self._obj_poses[i]
            radius = _obj[2] / 2
            found = False
            while not found:
                x = self._get_position(x_min, x_max, radius, wall_eps)
                if self._skewed:
                    y = skew_sigma * np.random.randn() + skew_mu
                    y = np.clip(y, radius + wall_eps, 1 - radius - wall_eps)
                else:
                    y = self._get_position(y_min, y_max, radius, wall_eps)
                found = True
                #for j in range(i):
                for j in range(objs.shape[0]):
                    if self._occlusion:
                        threshold = occlusion_threshold
                    else:
                        threshold = radius + objs[j, 2] / 2 + objs_eps
                    if norm(objs[j, 3:5] - [x, y]) < threshold:
                        found = False
                        break
                if self._agent_pos is not None:
                    if self._occlusion:
                        threshold = occlusion_threshold
                    else:
                        threshold = radius + objs[-1, 2] / 2 + agent_eps
                    if norm(objs[-1, 3:5] - [x, y]) < threshold:
                        found = False
            objs[i, 3] = x
            objs[i, 4] = y
        return objs

    def _set_objs(self):
        self._num_objects = np.random.choice(
            list(range(self._num_objs_range[0], self._num_objs_range[1] + 1))
        )

        if self._mode == "easy":
            if "Push" in self._name:
                assert self._num_objects == 3
                self._obj_poses = [
                    [0.25, 0.25, 0.75, 0.75],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.75, 0.75, 0.25, 0.25],
                ]
                self._agent_pos = [0.75, 0.75]
            else:
                if self._num_objects == 4:
                    self._obj_poses = [
                        #[0.25, 0.25, 0.75, 0.75],
                        #[0.25, 0.25, 0.25, 0.25],
                        #[0.75, 0.75, 0.25, 0.25],
                        #[0.75, 0.75, 0.75, 0.75],
                        [0.2, 0.3, 0.7, 0.8],
                        [0.2, 0.3, 0.2, 0.3],
                        [0.7, 0.8, 0.2, 0.3],
                        [0.7, 0.8, 0.7, 0.8],
                        #[0.15, 0.35, 0.65, 0.85],
                        #[0.15, 0.35, 0.15, 0.35],
                        #[0.65, 0.85, 0.15, 0.35],
                        #[0.65, 0.85, 0.65, 0.85],
                    ]
                elif self._num_objects == 2:
                    self._obj_poses = [
                        [0.15, 0.35, 0.65, 0.85],
                        [0.15, 0.35, 0.15, 0.35],
                    ]
                elif self._num_objects == 3:
                    self._obj_poses = [
                        [0.15, 0.35, 0.65, 0.85],
                        [0.15, 0.35, 0.15, 0.35],
                        [0.65, 0.85, 0.15, 0.35],
                    ]
                else:
                    raise NotImplemented
                self._agent_pos = [0.5, 0.5]
        elif self._mode == "normal":
            if "Push" in self._name:
                assert self._num_objects == 3
                self._obj_poses = [
                    [0.0, 0.5, 0.5, 1.0],
                    [0.0, 0.5, 0.0, 0.5],
                    [0.5, 1.0, 0.0, 0.5],
                ]
                self._agent_pos = [0.75, 0.75]
            else:
                assert self._num_objects == 4
                self._obj_poses = [
                    [0.0, 0.5, 0.5, 1.0],
                    [0.0, 0.5, 0.0, 0.5],
                    [0.5, 1.0, 0.0, 0.5],
                    [0.5, 1.0, 0.5, 1.0],
                ]
                self._agent_pos = [0.5, 0.5]
        elif self._mode == "hard":
            if "Push" in self._name:
                self._obj_poses = [[0.0, 1.0, 0.0, 1.0]] * (self._num_objects + 2)
            else:
                self._obj_poses = [[0.0, 1.0, 0.0, 1.0]] * (self._num_objects + 1)
        else:
            raise ValueError(
                f"{self._mode} is not supported, please select one of easy/normal/hard"
            )

        # color, shape, scale, x, y
        objs = np.zeros((self._num_objects + 1, 5), dtype=object)
        objs[-1, :3] = self._AGENT

        # target object index
        self._target_obj_idx = 0
        return objs

    def _get_masks(self, objs):
        masks = []
        bg = self._renderer.render([])
        for _obj in objs[:-1] if self._wo_agent else objs:
            rgb = [int(c * 255) for c in colors.to_rgb(_obj[0])]
            sprites= [
                Sprite(
                    _obj[3],
                    _obj[4],
                    _obj[1],
                    c0=rgb[0],
                    c1=rgb[1],
                    c2=rgb[2],
                    scale=_obj[2],
                )
            ]
            obs = self._renderer.render(sprites)
            obs = np.sum(np.abs(obs - bg), axis=-1)
            _mask = np.zeros((self._obs_size, self._obs_size, 1), dtype=int)
            _mask[obs!=0] = 1
            masks.append(_mask)
        fg_mask = np.sum(np.array(masks), axis=0)
        bg_mask = np.zeros((self._obs_size, self._obs_size, 1), dtype=int)
        bg_mask[fg_mask==0] = 1
        masks.append(bg_mask)
        return np.array(masks)

    def _draw_objs(self, objs, mode="rgb_array"):
        sprites = []
        for _obj in objs[:-1] if self._wo_agent else objs:
            if _obj[0] == -1:
                continue
            rgb = [int(c * 255) for c in colors.to_rgb(_obj[0])]
            sprites.append(
                Sprite(
                    _obj[3],
                    _obj[4],
                    _obj[1],
                    c0=rgb[0],
                    c1=rgb[1],
                    c2=rgb[2],
                    scale=_obj[2],
                )
            )
        obs = self._renderer.render(sprites)
        if mode == "rgb_array" or self._num_stacked_obss == 1:
            return obs
        else:
            self._stacked_obss.append(obs)
            return np.concatenate(
                self._stacked_obss[-1 * self._num_stacked_obss :], axis=-1
            )

    # distance between objs
    def _get_dist(self, i, j):
        return norm(self._objs[i, 3:5] - self._objs[j, 3:5])

    # Used for Target and Odd-one-out tasks
    def _cal_reward(self, reward, is_success, done):
        for i in range(self._num_objects):
            if norm(self._objs[i, 3:5] - self._objs[-1, 3:5]) < self._AGENT[2]:
                if i == self._target_obj_idx:
                    reward = 1.0
                    is_success = True
                else:
                    reward = 0.1 if self._rew_type == "normal" else 0.0
                    is_success = False
                done = True
                break
        return reward, is_success, done

    def reset(self):
        self._objs = self._set_objs()
        self.step_count = 0
        if self._use_bg:
            bg_img_name = self._bg_imgs[np.random.choice(len(self._bg_imgs))]
            if bg_img_name == "Black":
                bg_img = Image.new(
                    "RGB", (self._obs_size * 10, self._obs_size * 10), (0, 0, 0)
                )
            else:
                bg_img = Image.open(bg_img_name).resize(
                    (self._obs_size * 10, self._obs_size * 10)
                )
            self._renderer._canvas_bg = bg_img
        if self.render_mode == "state":
            self._stacked_obss = [
                np.zeros((self._num_objs_range[1] + 1, self._config.state_size))
            ] * (self._num_stacked_obss - 1)
        else:
            self._stacked_obss = [
                np.zeros((self._obs_size, self._obs_size, self._obs_channels))
            ] * (self._num_stacked_obss - 1)
        return self.render()

    def step(self, act):
        """
        act: {0,1,2,3} <- up, left, down, right
        """
        reward, done = 0.0, False
        dist_before_acting = self._get_dist(self._target_obj_idx, -1)
        # move
        if act == 0:
            self._objs[-1, 4] += self._moving_step_size
        elif act == 1:
            self._objs[-1, 3] -= self._moving_step_size
        elif act == 2:
            self._objs[-1, 4] -= self._moving_step_size
        elif act == 3:
            self._objs[-1, 3] += self._moving_step_size
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
            if self._get_dist(self._target_obj_idx, -1) < dist_before_acting:
                reward = 0.01
            else:
                reward = -0.01
        return reward, False, done

    def render(self, mode=None, fill_empty=True):
        if mode is None:
            mode = self.render_mode
        if mode == "state":
            gt_states = np.zeros(self._objs.shape)
            #for i in range(gt_states.shape[0] - 1):
            for i in range(gt_states.shape[0]):
                if self._objs[i, 0] == -1:
                    gt_states[i, :3] = -1
                    continue
                gt_states[i, 0] = COLORS.index(self._objs[i, 0])
                gt_states[i, 1] = SHAPES.index(self._objs[i, 1])
                gt_states[i, 2] = SCALES.index(self._objs[i, 2])
                gt_states[i, 3:] = self._objs[i, 3:]
            # Indexing Agent
            #gt_states[-1, 0] = COLORS.index(self._objs[-1, 0])
            #gt_states[-1, 1] = SHAPES.index(self._objs[-1, 1])
            #gt_states[-1, 2] = SCALES.index(self._objs[-1, 2])
            #gt_states[-1, 3:] = self._objs[-1, 3:]
            #gt_states[:,3:] = gt_states[:,3:] * 64
            gt_states = np.array(gt_states, dtype=np.float32)
            if fill_empty:
                zero_padding_size = self._num_objs_range[1] + 1 - gt_states.shape[0]
                if zero_padding_size > 0:
                    zero_padding = np.zeros((zero_padding_size, self._config.state_size))
                    gt_states = np.concatenate([gt_states, zero_padding], axis=0)
            if self._num_stacked_obss == 1:
                return gt_states
            else:
                self._stacked_obss.append(gt_states)
                return np.concatenate(
                    self._stacked_obss[-1 * self._num_stacked_obss :], axis=-1
                )
        elif mode == "mask":
            masks = self._get_masks(self._objs) # objs, agent and bg
            if fill_empty:
                zero_padding_size = self._num_objs_range[1] + 2 - masks.shape[0]
                if zero_padding_size > 0:
                    zero_padding = np.zeros((zero_padding_size, self._obs_size, self._obs_size, 1))
                    masks = np.concatenate([masks[:-1], zero_padding, masks[-1:]], axis=0)
            return masks
        else:
            return self._draw_objs(self._objs, mode)

    def close(self):
        self._objs = None
        self.step_count = 0
