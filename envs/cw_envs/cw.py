import inspect
import os

import gym
import numpy as np
import pybullet
from causal_world.envs import CausalWorld
from causal_world.envs.robot.camera import Camera
from causal_world.envs.robot.trifinger import TriFingerRobot
from causal_world.envs.scene.stage import Stage
from causal_world.loggers.tracker import Tracker
from causal_world.task_generators.task import generate_task


# Taken from CausalWorld superclass and subclassed so we can
# show silhoutte images (goals) with the finger cameras
class MyCausalWorld(CausalWorld):
    def __init__(
        self,
        task=None,
        skip_frame=10,
        enable_visualization=False,
        seed=0,
        action_mode="joint_positions",
        observation_mode="structured",
        normalize_actions=True,
        normalize_observations=True,
        max_episode_length=None,
        data_recorder=None,
        camera_indicies=np.array([0, 1, 2]),
        wrappers=None,
    ):
        """
        The causal world encapsulates the environment of the agent, where you
        can perform actions, intervene, reset the state..etc

        :param task: (causal_world.BaseTask) this is the task produced by one of
                                            the available task generators or the
                                            custom task created.
        :param skip_frame: (int) the low level controller is running @250Hz
                           which corresponds to skip frame of 1, a skip frame
                           of 250 corresponds to frequency of 1Hz
        :param enable_visualization: (bool) this is a boolean which indicates
                                     if a GUI is enabled or the environment
                                     should operate in a headless mode.
        :param seed: (int) this is the random seed used in the environment.
        :param action_mode: (str) action_modes available are "joint_positions",
                                     "end_effector_positions" which is non
                                     deterministic at the moment since it uses
                                     IK of pybullet and lastly "joint_torques".
        :param observation_mode: (str) observation modes available are
                                          "structured" or "pixel" modes.
                                          The structured observations are
                                          specified in the task generator
                                          itself. For the "pixel" mode
                                          you will get maximum 6 images
                                          concatenated where the first half
                                          are the current images rendered in
                                          the platform and the second half are
                                          the goal images rendered from the
                                          same point of view.
        :param normalize_actions: (bool) this is a boolean which specifies
                                         whether the actions passed to the step
                                         function are normalized or not.
        :param normalize_observations: (bool) this is a boolean which specifies
                                              whether the observations returned
                                              should be normalized or not.
        :param max_episode_length: (int) it specifies what is the episode length
                                         of the task, by default this will be
                                         calculated according to how many
                                         objects are in the arena
                                         (5 secs * number of objects).
        :param data_recorder: (causal_world.DataRecorder) passed only when
                                                            you want to log and
                                                            record the episodes
                                                            which can be used
                                                            further in imitation
                                                            learning for instance.
        :param camera_indicies: (list) maximum of 3 elements where each element
                                       is from 0 to , specifies which cameras
                                       to return in the observations and the
                                       order as well.
        :param wrappers: (causal_world.wrappers) should not be used for now.
        """
        self._observation_mode = observation_mode
        self._action_mode = action_mode
        self._enable_visualization = enable_visualization
        self.seed(seed)
        self._simulation_time = 1.0 / 250
        self._camera_indicies = np.array(camera_indicies)
        self._skip_frame = skip_frame
        self.dt = self._simulation_time * self._skip_frame
        self._pybullet_client_w_o_goal_id = None
        self._pybullet_client_w_goal_id = None
        self._pybullet_client_full_id = None
        self._revolute_joint_ids = None
        self._instantiate_pybullet()
        self.link_name_to_index = None
        self._robot_properties_path = os.path.join(
            os.path.dirname(inspect.getmodule(super()).__file__),
            "../assets/robot_properties_fingers",
        )
        self._finger_urdf_path = os.path.join(
            self._robot_properties_path, "urdf", "trifinger_edu.urdf"
        )
        self._create_world(initialize_goal_image=True)
        self._tool_cameras = None
        self._goal_cameras = None
        if observation_mode == "pixel":
            self._tool_cameras = []
            self._tool_cameras.append(
                Camera(
                    camera_position=[0.2496, 0.2458, 0.58],
                    camera_orientation=[0.3760, 0.8690, -0.2918, -0.1354],
                    pybullet_client_id=self._pybullet_client_full_id,
                )
            )
            self._tool_cameras.append(
                Camera(
                    camera_position=[0.0047, -0.2834, 0.58],
                    camera_orientation=[0.9655, -0.0098, -0.0065, -0.2603],
                    pybullet_client_id=self._pybullet_client_full_id,
                )
            )
            self._tool_cameras.append(
                Camera(
                    camera_position=[-0.2470, 0.2513, 0.50],
                    camera_orientation=[-0.3633, 0.8686, -0.3141, 0.1220],
                    pybullet_client_id=self._pybullet_client_full_id,
                )
            )
            self._goal_cameras = []
            self._goal_cameras.append(
                Camera(
                    camera_position=[0.2496, 0.2458, 0.58],
                    camera_orientation=[0.3760, 0.8690, -0.2918, -0.1354],
                    pybullet_client_id=self._pybullet_client_w_goal_id,
                )
            )
            self._goal_cameras.append(
                Camera(
                    camera_position=[0.0047, -0.2834, 0.58],
                    camera_orientation=[0.9655, -0.0098, -0.0065, -0.2603],
                    pybullet_client_id=self._pybullet_client_w_goal_id,
                )
            )
            self._goal_cameras.append(
                Camera(
                    camera_position=[-0.2470, 0.2513, 0.50],
                    camera_orientation=[-0.3633, 0.8686, -0.3141, 0.1220],
                    pybullet_client_id=self._pybullet_client_w_goal_id,
                )
            )
        self._robot = TriFingerRobot(
            action_mode=action_mode,
            observation_mode=observation_mode,
            skip_frame=skip_frame,
            normalize_actions=normalize_actions,
            normalize_observations=normalize_observations,
            simulation_time=self._simulation_time,
            pybullet_client_full_id=self._pybullet_client_full_id,
            pybullet_client_w_goal_id=self._pybullet_client_w_goal_id,
            pybullet_client_w_o_goal_id=self._pybullet_client_w_o_goal_id,
            revolute_joint_ids=self._revolute_joint_ids,
            finger_tip_ids=self.finger_tip_ids,
            cameras=self._tool_cameras,
            camera_indicies=self._camera_indicies,
        )
        self._stage = Stage(
            observation_mode=observation_mode,
            normalize_observations=normalize_observations,
            pybullet_client_full_id=self._pybullet_client_full_id,
            pybullet_client_w_goal_id=self._pybullet_client_w_goal_id,
            pybullet_client_w_o_goal_id=self._pybullet_client_w_o_goal_id,
            cameras=self._goal_cameras,
            camera_indicies=self._camera_indicies,
        )
        gym.Env.__init__(self)
        if task is None:
            self._task = generate_task("reaching")
        else:
            self._task = task
        self._task.init_task(
            self._robot, self._stage, max_episode_length, self._create_world
        )
        if max_episode_length is None:
            max_episode_length = int(task.get_default_max_episode_length() / self.dt)
        self._max_episode_length = max_episode_length
        self._reset_observations_space()
        self.action_space = self._robot.get_action_spaces()

        self.metadata["video.frames_per_second"] = (
            1 / self._simulation_time
        ) / self._skip_frame
        self._setup_viewing_camera()
        self._normalize_actions = normalize_actions
        self._normalize_observations = normalize_observations
        self._episode_length = 0
        self._data_recorder = data_recorder
        self._wrappers_dict = dict()
        self._tracker = Tracker(task=self._task, world_params=self.get_world_params())
        self._scale_reward_by_dt = True
        self._disabled_actions = False
        return

    def render(self, mode="human"):
        """
        Returns an RGB image taken from above the platform.

        :param mode: (str) not taken in account now.

        :return: (nd.array) an RGB image taken from above the platform.
        """
        client = self._pybullet_client_full_id
        (_, _, px, _, _) = pybullet.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client)
        rgb_array = np.array(px)
        if rgb_array.ndim == 1:
            rgb_array = rgb_array.reshape((self._render_height, self._render_width, 4))
        rgb_array = np.asarray(rgb_array, dtype='uint8')
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


    def _instantiate_pybullet(self):
        """
        This function is used for instantiating all pybullet instances used for
        the current simulation

        :return:
        """
        if self._observation_mode == "pixel":
            self._pybullet_client_w_o_goal_id = pybullet.connect(pybullet.DIRECT)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI,
                0,
                physicsClientId=self._pybullet_client_w_o_goal_id,
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_w_o_goal_id,
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_w_o_goal_id,
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_w_o_goal_id,
            )
            pybullet.resetSimulation(physicsClientId=self._pybullet_client_w_o_goal_id)
            pybullet.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self._pybullet_client_w_o_goal_id,
            )
            self._pybullet_client_w_goal_id = pybullet.connect(pybullet.DIRECT)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI,
                0,
                physicsClientId=self._pybullet_client_w_goal_id,
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_w_goal_id,
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_w_goal_id,
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_w_goal_id,
            )
            pybullet.resetSimulation(physicsClientId=self._pybullet_client_w_goal_id)
            pybullet.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self._pybullet_client_w_goal_id,
            )
            if self._enable_visualization:
                self._pybullet_client_full_id = pybullet.connect(pybullet.GUI)
            else:
                self._pybullet_client_full_id = pybullet.connect(pybullet.DIRECT)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI,
                0,
                physicsClientId=self._pybullet_client_full_id,
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_full_id,
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_full_id,
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_full_id,
            )
            pybullet.resetSimulation(physicsClientId=self._pybullet_client_full_id)
            pybullet.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self._pybullet_client_full_id,
            )
        else:
            if self._enable_visualization:
                self._pybullet_client_full_id = pybullet.connect(pybullet.GUI)
            else:
                self._pybullet_client_full_id = pybullet.connect(pybullet.DIRECT)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI,
                0,
                physicsClientId=self._pybullet_client_full_id,
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_full_id,
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_full_id,
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_full_id,
            )
            pybullet.resetSimulation(physicsClientId=self._pybullet_client_full_id)
            pybullet.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self._pybullet_client_full_id,
            )
        return
