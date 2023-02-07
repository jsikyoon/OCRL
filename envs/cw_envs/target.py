import gym
import numpy as np
from causal_world.configs.world_constants import WorldConstants
from causal_world.envs import CausalWorld
from causal_world.task_generators.base_task import BaseTask
from causal_world.utils.rotation_utils import cart2cyl
from matplotlib import colors
from PIL import Image

from .cw import MyCausalWorld


def CwTargetEnv(config, seed):
    np.random.seed(seed)
    assert config.mode in ["easy", "hard"]  # no normal for now
    assert config.rew_type in ["sparse"]  # only sparse for now
    task = SingleFingerReachTask(activate_sparse_reward=True)
    if config.render_mode == "finger_image":
        env = MyCausalWorld(
            seed=seed,
            task=task,
            observation_mode="pixel",
            camera_indicies=[0, 1, 2],
            skip_frame=10,
            enable_visualization=False,
        )
        env = SingleFingerCausalWorldWrapper(env, config)
        env = CausalWorldFingerImageWrapper(env, config)
    else:
        env = CausalWorld(
            seed=seed,
            task=task,
            observation_mode="structured",
            camera_indicies=[0],
            skip_frame=10,
            enable_visualization=False,
        )
        env = SingleFingerCausalWorldWrapper(env, config)
        if config.render_mode == "state":
            # hack for better performance so we don't need to render images
            env = CausalRLStateOnlyWrapper(env)
        else:
            env = CausalRLRenderAndStateWrapper(env)
        if config.render_mode == "image":
            obs_key = "image"
        elif config.render_mode == "state":
            obs_key = "gt"
        env = SelectObsKeyWrapper(env, obs_key=obs_key)

    return env


class CausalWorldFingerImageWrapper(gym.Wrapper):
    def __init__(self, env, config, height=64, width=64):
        super().__init__(env)
        self.env = env
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, config.num_stacked_obss * 3),
            dtype=np.uint8,
        )

    def _get_frame(self, obs):
        frames = []
        # only take the first half of images because second half are goal images
        for i in range(len(obs) // 2):
            frame = Image.fromarray((obs[i] * 255).astype(np.uint8))
            frame = frame.resize((self.height, self.width), Image.BILINEAR)
            frame = np.asarray(frame)
            frame = frame.astype(np.uint8)
            frames.append(frame)
        frames = np.stack(frames)
        frames = frames.transpose(1, 2, 0, 3).reshape(self.height, self.width, -1)
        return frames

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self._get_frame(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward > 0:
            info["is_success"] = True
        else:
            info["is_success"] = False

        return self._get_frame(obs), reward, done, info


class SingleFingerCausalWorldWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.env = env
        self._config = config
        self._COLORS = self._config.COLORS
        if len(self._config.target) > 0:
            self._target_color = self._config.target[0]
        else:
            self._target_color = np.random.choice(self._config.COLORS)
        self._size = [0.055, 0.055, 0.055]
        action_space_shape = (3,)
        self._JOINTS_RAISED_POSITIONS = [
            -1.56,
            -0.08,
            -2.7,
            -1.56,
            -0.08,
            -2.7,
            -1.56,
            -0.08,
            -2.7,
        ]
        self._JOINTS_RAISED_ACTION = [
            -1.56,
            -0.08,
            -2.7,
        ]

        # these need to be float32 for sb3
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=action_space_shape,
            dtype=np.float32,
        )
        self.num_objects = 4
        self.target_obj_idx = None

    def _get_random_positions(self):
        cube_size = 0.065

        def _pair_has_collision(p1, p2):
            if p1[0] > p2[0] + cube_size:
                return False
            if p1[0] < p2[0] - cube_size:
                return False
            if p1[1] > p2[1] + cube_size:
                return False
            if p1[1] < p2[1] - cube_size:
                return False
            return True

        def _check_collision(new_position, old_positions):
            for p in old_positions:
                if _pair_has_collision(new_position, p):
                    return True
            return False

        cart_positions = []
        for _ in range(self.num_objects):
            while True:
                cart = self.env._task._stage.random_position(
                    height_limits=(0.0325, 0.0325), angle_limits=(-3.14, 3.14 / 4)
                )
                if not _check_collision(cart, cart_positions):
                    break
            cart_positions.append(cart)
        return [cart2cyl(p) for p in cart_positions]

    def reset(self, **kwargs):
        obs = self.env.reset()
        interventions = {}
        interventions["joint_positions"] = self._JOINTS_RAISED_POSITIONS
        self.target_obj_idx = np.random.randint(self.num_objects)
        self.env._task.target_obj = f"obj_{self.target_obj_idx}"
        if self._config.mode == "easy":
            positions = [
                [0.14, -1.0, 0.0325],
                [0.15, -2.34, 0.0325],
                [0.15, 0.01, 0.0325],
                [0.03, -3.14, 0.0325],
            ]
        elif self._config.mode == "hard":
            positions = self._get_random_positions()
        else:
            raise NotImplementedError()

        if self._config.task == "target":
            for n_idx in range(self.num_objects):
                if n_idx == self.target_obj_idx:
                    color = self._target_color
                else:
                    found = False
                    while not found:
                        color = np.random.choice(self._COLORS)
                        found = color != self._target_color

                interventions[f"obj_{n_idx}"] = {
                    "color": colors.to_rgb(color),
                    "cylindrical_position": positions[n_idx],
                    "size": np.asarray(self._size),
                }
        if self._config.task == "ooo":
            self._target_color = np.random.choice(self._config.COLORS)
            other_color = None
            while other_color is None:
                candidate_color = np.random.choice(self._COLORS)
                if candidate_color != self._target_color:
                    other_color = candidate_color
            for n_idx in range(self.num_objects):
                if n_idx == self.target_obj_idx:
                    color = self._target_color
                else:
                    color = other_color

                interventions[f"obj_{n_idx}"] = {
                    "color": colors.to_rgb(color),
                    "cylindrical_position": positions[n_idx],
                    "size": np.asarray(self._size),
                }

        success_signal, obs = self.env.do_intervention(interventions)
        return obs

    def step(self, action):
        raised_action = self._JOINTS_RAISED_ACTION
        action = np.asarray(
            raised_action + list(action) + raised_action, dtype=np.float32
        )

        return self.env.step(action)


class SelectObsKeyWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_key):
        super().__init__(env)
        self.obs_key = obs_key
        self.observation_space = env.observation_space[self.obs_key]

    def observation(self, obs):
        return obs[self.obs_key]


class CausalRLRenderWrapper(gym.Wrapper):
    def __init__(self, env, height=64, width=64):
        super().__init__(env)
        self.env = env
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

    def _get_frame(self, obs):
        frame = self.env.render()
        frame = Image.fromarray((frame).astype(np.uint8))
        frame = frame.resize((self.height, self.width), Image.BILINEAR)
        frame = np.asarray(frame)
        # frame = frame.transpose(2, 0, 1)
        frame = frame.astype(np.uint8)
        return frame

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self._get_frame(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward > 0:
            info["is_success"] = True
        else:
            info["is_success"] = False

        return self._get_frame(obs), reward, done, info

class CausalRLStateOnlyWrapper(gym.Wrapper):
    def __init__(self, env, height=64, width=64):
        super().__init__(env)
        self.env = env
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Dict(
            {
                "robot_state": gym.spaces.Box(
                    low=-1, high=1, shape=(28,), dtype=np.float
                ),
                "object_states": gym.spaces.Box(
                    low=-1, high=1, shape=(40,), dtype=np.float
                ),
                "gt": gym.spaces.Box(low=-1, high=1, shape=(5, 40), dtype=np.float),
            }
        )

    def _get_frame(self, obs):
        # first variable is time left
        time_left = obs[0]

        # 9 joint positions, 9 joint velocities, 9 end effector positions
        # include time_left for now
        robot_state = obs[0:28]

        # 4 each of (cartesian position (3), type (1), size (3), color (3))
        # for a total of 4 * 10 = 40
        object_states = obs[28:]

        # gt state consists of robot state and object states. Make them all the same size
        # 5 -> first robot state + 1 objects

        # we also add one dimension indicating the object type (0=robot arm, 1=block)
        gt = np.zeros((5, 40))
        gt[0][:28] = robot_state
        gt[0][-1] = 0
        for i in range(4):
            gt[i + 1][28:38] = object_states[i * 10 : (i * 10) + 10]
            gt[i+1][31] = 1
            gt[i + 1][-1] = 1

        return {
            "robot_state": robot_state,
            "object_states": object_states,
            "gt": gt,
        }

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self._get_frame(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward > 0:
            info["is_success"] = True
        else:
            info["is_success"] = False
        return self._get_frame(obs), reward, done, info




class CausalRLRenderAndStateWrapper(gym.Wrapper):
    def __init__(self, env, height=64, width=64):
        super().__init__(env)
        self.env = env
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
                ),
                "robot_state": gym.spaces.Box(
                    low=-1, high=1, shape=(28,), dtype=np.float
                ),
                "object_states": gym.spaces.Box(
                    low=-1, high=1, shape=(40,), dtype=np.float
                ),
                "gt": gym.spaces.Box(low=-1, high=1, shape=(5, 28), dtype=np.float),
            }
        )

    def _get_frame(self, obs):
        frame = self.env.render()
        frame = Image.fromarray((frame).astype(np.uint8))
        frame = frame.resize((self.height, self.width), Image.BILINEAR)
        frame = np.asarray(frame)
        # frame = frame.transpose(2, 0, 1)
        frame = frame.astype(np.uint8)

        # first variable is time left
        time_left = obs[0]

        # 9 joint positions, 9 joint velocities, 9 end effector positions
        # include time_left for now
        robot_state = obs[0:28]

        # 4 each of (cartesian position (3), type (1), size (3), color (3))
        # for a total of 4 * 10 = 40
        object_states = obs[28:]

        # gt state consists of robot state and object states. Make them all the same size
        # 5 -> first robot state + 1 objects
        gt = np.zeros((5, 28))
        gt[0] = robot_state
        for i in range(4):
            gt[i + 1][:10] = object_states[i * 10 : (i * 10) + 10]

        return {
            "image": frame,
            "robot_state": robot_state,
            "object_states": object_states,
            "gt": gt,
        }

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self._get_frame(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward > 0:
            info["is_success"] = True
        else:
            info["is_success"] = False
        return self._get_frame(obs), reward, done, info


class SingleFingerReachTask(BaseTask):
    def __init__(
        self,
        variables_space="space_a_b",
        fractional_reward_weight=1,
        dense_reward_weights=np.array([100000, 0, 0, 0]),
        joint_positions=None,
        activate_sparse_reward=False,
    ):
        """
        This task generator will generate a task for reaching.

         :param variables_space: (str) space to be used either 'space_a' or
                                      'space_b' or 'space_a_b'
        :param fractional_reward_weight: (float) weight multiplied by the
                                                fractional volumetric
                                                overlap in the reward.
        :param dense_reward_weights: (list float) specifies the reward weights
                                                  for all the other reward
                                                  terms calculated in the
                                                  calculate_dense_rewards
                                                  function.
        :param joint_positions: (nd.array) specifies the joints position to start
                                            the episode with. None if the default
                                            to be used.
        :param activate_sparse_reward: (bool) specified if you want to
                                              sparsify the reward by having
                                              +1 or 0 if the mean distance
                                              from goal is < 0.01.
        """
        super().__init__(
            task_name="reaching",
            variables_space=variables_space,
            fractional_reward_weight=fractional_reward_weight,
            dense_reward_weights=dense_reward_weights,
            activate_sparse_reward=activate_sparse_reward,
        )
        self._task_robot_observation_keys = [
            "time_left_for_task",
            "joint_positions",
            "joint_velocities",
            "end_effector_positions",
        ]
        # self._task_params["joint_positions"] = joint_positions
        self._task_params["joint_positions"] = [
            -1.56,
            -0.08,
            -2.7,
            -1.56,
            -0.08,
            -2.7,
            -1.56,
            -0.08,
            -2.7,
        ]

        self.previous_end_effector_positions = None
        self.previous_joint_velocities = None
        self.current_number_of_obstacles = 0
        self.target_obj = None
        self._finger_idx = 1
        self._reach_threshold = 0.021

    def _set_up_stage_arena(self):
        """

        :return:
        """
        creation_dict = {
            "name": "obj_0",
            "shape": "cube",
            "color": np.array([1, 0, 0]),
            "position": [0.0, -0.15, 0],
        }
        self._stage.add_silhoutte_general_object(**creation_dict)
        creation_dict = {
            "name": "obj_1",
            "shape": "cube",
            "color": np.array([0, 1, 0]),
            "position": [0.2, 0, 0],
        }
        self._stage.add_silhoutte_general_object(**creation_dict)
        creation_dict = {
            "name": "obj_2",
            "shape": "cube",
            "color": np.array([0, 0, 1]),
            "position": [-0.2, 0, 0],
        }
        self._stage.add_silhoutte_general_object(**creation_dict)
        creation_dict = {
            "name": "obj_3",
            "shape": "cube",
            "color": np.array([0, 0, 1]),
            "position": [0, 0, 0],
        }
        self._stage.add_silhoutte_general_object(**creation_dict)

        self._task_stage_observation_keys = [
            "obj_0_cartesian_position",
            "obj_0_type",
            "obj_0_size",
            "obj_0_color",
            "obj_1_cartesian_position",
            "obj_1_type",
            "obj_1_size",
            "obj_1_color",
            "obj_2_cartesian_position",
            "obj_2_type",
            "obj_2_size",
            "obj_2_color",
            "obj_3_cartesian_position",
            "obj_3_type",
            "obj_3_size",
            "obj_3_color",
        ]
        return

    def get_description(self):
        """

        :return: (str) returns the description of the task itself.
        """
        return "Task where the goal is to reach a " "goal point for each finger"

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        """

        :param desired_goal:
        :param achieved_goal:

        :return:
        """
        end_effector_positions_goal = desired_goal
        current_end_effector_positions = achieved_goal[
            self._finger_idx * 3 : (self._finger_idx * 3) + 3
        ]
        previous_end_effector_positions = self.previous_end_effector_positions[
            self._finger_idx * 3 : (self._finger_idx * 3) + 3
        ]

        previous_dist_to_goal = np.linalg.norm(
            end_effector_positions_goal - previous_end_effector_positions
        )
        current_dist_to_goal = np.linalg.norm(
            end_effector_positions_goal - current_end_effector_positions
        )
        rewards = list()
        rewards.append(previous_dist_to_goal - current_dist_to_goal)
        rewards.append(-current_dist_to_goal)
        rewards.append(-np.linalg.norm(self._robot.get_latest_full_state()["torques"]))
        rewards.append(
            -np.linalg.norm(
                np.abs(
                    self._robot.get_latest_full_state()["velocities"][
                        self._finger_idx * 3 : (self._finger_idx * 3) + 3
                    ]
                    - previous_end_effector_positions
                ),
                ord=2,
            )
        )
        update_task_info = {
            "current_end_effector_positions": achieved_goal,
            "current_velocity": self._robot.get_latest_full_state()["velocities"],
        }
        return rewards, update_task_info

    def _update_task_state(self, update_task_info):
        """

        :param update_task_info:

        :return:
        """
        self.previous_end_effector_positions = update_task_info[
            "current_end_effector_positions"
        ]
        self.previous_joint_velocities = update_task_info["current_velocity"]
        return

    def _set_task_state(self):
        """

        :return:
        """
        self.previous_end_effector_positions = self._robot.get_latest_full_state()[
            "end_effector_positions"
        ]
        self.previous_joint_velocities = self._robot.get_latest_full_state()[
            "velocities"
        ]
        return

    def get_desired_goal(self):
        """

        :return: (nd.array) specifies the desired goal as array of all three
                            positions of the finger goals.
        """
        desired_goal = np.array([])
        if self.target_obj is not None:
            desired_goal = np.append(
                desired_goal,
                self._stage.get_object_state(self.target_obj, "cartesian_position"),
            )
        return desired_goal

    def is_done(self):
        # TODO: dynamic number of objects?
        end_effector_positions = self.get_achieved_goal()
        for idx in range(4):
            obj_pos = (
                self._stage.get_object_state(f"obj_{idx}", "cartesian_position"),
            )
            dist_to_obj = self._goal_reward(end_effector_positions, obj_pos)
            if self._check_preliminary_success(dist_to_obj):
                return True

        return False

    def get_achieved_goal(self):
        """

        :return: (nd.array) specifies the achieved goal as concatenated
                            end-effector positions.
        """
        achieved_goal = self._robot.get_latest_full_state()["end_effector_positions"]
        return np.array(achieved_goal)

    def _goal_reward(self, achieved_goal, desired_goal):
        """

        :param achieved_goal:
        :param desired_goal:

        :return:
        """
        current_end_effector_positions = achieved_goal[
            self._finger_idx * 3 : (self._finger_idx * 3) + 3
        ]
        current_dist_to_goal = np.abs(desired_goal - current_end_effector_positions)
        current_dist_to_goal_mean = np.mean(current_dist_to_goal)
        return np.array(current_dist_to_goal_mean)

    def _check_preliminary_success(self, goal_reward):
        """

        :param goal_reward:

        :return:
        """
        if goal_reward < self._reach_threshold:
            return True
        else:
            return False

    def _calculate_fractional_success(self, goal_reward):
        """

        :param goal_reward:
        :return:
        """
        clipped_distance = np.clip(goal_reward, 0.01, 0.03)
        distance_from_success = clipped_distance - 0.01
        fractional_success = 1 - (distance_from_success / 0.02)
        return fractional_success

    def get_info(self):
        """

        :return: (dict) returns the info dictionary after every step of the
                        environment.
        """
        info = dict()
        info["desired_goal"] = self._current_desired_goal
        info["achieved_goal"] = self._current_achieved_goal
        info["success"] = self._task_solved
        if self._is_ground_truth_state_exposed:
            info[
                "ground_truth_current_state_varibales"
            ] = self.get_current_variable_values()
        if self._is_partial_solution_exposed:
            info["possible_solution_intervention"] = dict()
            info["possible_solution_intervention"][
                "joint_positions"
            ] = self._robot.get_joint_positions_from_tip_positions(
                self._current_desired_goal,
                self._robot.get_latest_full_state()["positions"],
            )
        info["fractional_success"] = self._calculate_fractional_success(
            self._current_goal_reward
        )
        return info

    def _set_intervention_space_a(self):
        """

        :return:
        """
        super(SingleFingerReachTask, self)._set_intervention_space_a()
        self._intervention_space_a["number_of_obstacles"] = np.array([1, 5])

        return

    def _set_intervention_space_b(self):
        """

        :return:
        """
        super(SingleFingerReachTask, self)._set_intervention_space_b()
        self._intervention_space_b["number_of_obstacles"] = np.array([1, 5])
        return

    def get_task_generator_variables_values(self):
        """

        :return: (dict) specifying the variables belonging to the task itself.
        """
        task_generator_variables = dict()
        task_generator_variables[
            "number_of_obstacles"
        ] = self.current_number_of_obstacles
        return task_generator_variables

    def apply_task_generator_interventions(self, interventions_dict):
        """

        :param interventions_dict: (dict) variables and their corresponding
                                   intervention value.

        :return: (tuple) first position if the intervention was successful or
                         not, and second position indicates if
                         observation_space needs to be reset.
        """
        if len(interventions_dict) == 0:
            return True, False
        reset_observation_space = False
        if "number_of_obstacles" in interventions_dict:
            if (
                int(interventions_dict["number_of_obstacles"])
                > self.current_number_of_obstacles
            ):
                reset_observation_space = True
                for i in range(
                    self.current_number_of_obstacles,
                    int(interventions_dict["number_of_obstacles"]),
                ):
                    self._stage.add_rigid_general_object(
                        name="obstacle_" + str(i),
                        shape="static_cube",
                        size=np.array([0.01, 0.01, 0.01]),
                        color=np.array([0, 0, 0]),
                        position=np.random.uniform(
                            WorldConstants.ARENA_BB[0], WorldConstants.ARENA_BB[1]
                        ),
                    )
                    self.current_number_of_obstacles += 1
                    self._task_stage_observation_keys.append(
                        "obstacle_" + str(i) + "_type"
                    )
                    self._task_stage_observation_keys.append(
                        "obstacle_" + str(i) + "_size"
                    )
                    self._task_stage_observation_keys.append(
                        "obstacle_" + str(i) + "_cartesian_position"
                    )
                    self._task_stage_observation_keys.append(
                        "obstacle_" + str(i) + "_orientation"
                    )
            else:
                return True, reset_observation_space
        else:
            raise Exception("this task generator variable " "is not yet defined")
        self._set_intervention_space_b()
        self._set_intervention_space_a()
        self._set_intervention_space_a_b()
        self._stage.finalize_stage()
        return True, reset_observation_space
