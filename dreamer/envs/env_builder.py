# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gym
import numpy as np
import envs.utilities.a1_randomizer_ground as a1_rg
from robots import robot_config
from robots import a1
from envs.sensors import robot_sensors
from envs.sensors import sensor_wrappers
from envs.sensors import environment_sensors
from envs.env_wrappers import move_forward_task
from envs.env_wrappers import move_forward_task_mpc
from envs.env_wrappers import goal_task
from envs.env_wrappers import observation_dictionary_to_array_wrapper
from envs.env_wrappers import curriculum_wrapper_env
from envs.utilities import controllable_env_randomizer_from_config
from envs import locomotion_gym_config
from envs import locomotion_gym_env_with_rich_information
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
  inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class ActionRestrain(gym.ActionWrapper):
  # Current for POSITION only
  def __init__(self, env, clip_num):
    super().__init__(env)

    self.base_angle = np.array(list(a1.INIT_MOTOR_ANGLES))
    self.clip_num = clip_num
    if isinstance(self.clip_num, list):
      self.clip_num = np.array(self.clip_num)
      assert len(clip_num) == np.prod(self.base_angle.shape)

    self.ub = self.base_angle + self.clip_num
    self.lb = self.base_angle - self.clip_num
    self.action_space = gym.spaces.Box(self.lb, self.ub)

  def action(self, action):
    return np.clip(action, self.lb, self.ub)


class MPCActionRestrain(gym.ActionWrapper):

  def __init__(self, env, clip_num):
    super().__init__(env)

    self.clip_num = clip_num
    if isinstance(self.clip_num, list):
      self.clip_num = np.array(self.clip_num)
      assert len(clip_num) == 2

    self.ub = self.clip_num
    self.lb = - self.clip_num
    self.action_space = gym.spaces.Box(self.lb, self.ub)

  def action(self, action):
    return np.clip(action, self.lb, self.ub)


class ActionResidual(gym.ActionWrapper):
  # Current for POSITION only
  def __init__(self, env, clip_num):
    super().__init__(env)
    self.clip_num = clip_num
    self.base_angle = np.array(a1.INIT_MOTOR_ANGLES)
    self.ub = np.ones_like(self.base_angle) * self.clip_num
    self.lb = -self.ub
    self.action_space = gym.spaces.Box(self.lb, self.ub)

  def action(self, action):
    current_angles = self.robot.GetMotorAngles()
    biased_action = np.clip(action,
                            self.lb, self.ub
                            ) + current_angles
    return biased_action


class DiagonalAction(gym.ActionWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.lb = np.split(self.env.action_space.low, 2)[0]
    self.ub = np.split(self.env.action_space.high, 2)[0]
    self.action_space = gym.spaces.Box(self.lb, self.ub)

  def action(self, action):
    right_act, left_act = np.split(action, 2)
    act = np.concatenate(
      [right_act, left_act, left_act, right_act]
    )
    return act


class RandoDirWrapper(gym.ObservationWrapper):
  def __init__(self, env, dir_update_interval=None):
    super().__init__(env)

    self.observation_space = gym.spaces.Box(
      np.concatenate([[0, 0], self.env.observation_space.low]),
      np.concatenate([[1, 1], self.env.observation_space.high])
    )
    self.current_angle = 0
    self.current_vec = np.array([
      np.cos(self.current_angle),
      np.sin(self.current_angle)
    ])
    self.dir_update_interval = dir_update_interval
    self.time_count_randdir = 0

  def observation(self, observation):
    self.time_count_randdir += 1
    if self.dir_update_interval is not None and \
        self.time_count_randdir % self.dir_update_interval == 0:

      self.current_angle = np.random.uniform(
        low=-np.pi / 2,
        high=np.pi / 2
      )
      self.current_vec = np.array([
        np.cos(self.current_angle),
        np.sin(self.current_angle)
      ])
      self.env.task.target_vel_dir = self.current_vec

    obs = np.concatenate([self.current_vec, observation])

    return obs

  def reset(self):
    self.time_count_randdir = 0
    self.current_angle = np.random.uniform(
      low=-np.pi / 2,
      high=np.pi / 2
    )
    self.current_vec = np.array([
      np.cos(self.current_angle),
      np.sin(self.current_angle)
    ])
    self.env.task.target_vel_dir = self.current_vec
    return super().reset()


def build_a1_ground_env(
    motor_control_mode="POSITION",
    z_constrain=False,
    other_direction_penalty=0,
    z_penalty=1,
    clip_num=[0.05, 0.5, 0.5, 0.05, 0.5, 0.5, 0.05, 0.5, 0.5, 0.05, 0.5, 0.5],
    enable_rendering=False,
    diagonal_act=False,
    num_action_repeat=16,
    time_step_s=0.0025,
    add_last_action_input=True,
    enable_action_interpolation=False,
    enable_action_filter=False,
    domain_randomization=True,
    get_image=True,
    depth_image=False,
    depth_norm=False,
    grayscale=True,
    rgbd=False,
    fric_coeff=[0.8, 0.1, 0.1],
    terrain_type="random_blocks_sparse",
    alive_reward=0.1,
    fall_reward=0.0,
    target_vel=1,
    random_init_range=1.0,
    dir_update_interval=None,
    check_contact=False,
    random_dir=False,
    rotate_sensor=False,
    frame_extract=1,
    goal=False,
    subgoal=False,
    goal_coeff=10,
    subgoal_reward=None,
    record_video=False,
    no_displacement=False,
    get_image_interval=1,
    reset_frame_idx=False,
    reset_frame_idx_each_step=False,
    random_shape=True,
    moving=False,
    curriculum=False,
    interpolation=False,
    fixed_delay_observation=False,
):
  gym_config = locomotion_gym_config.LocomotionGymConfig(
    simulation_parameters=locomotion_gym_config.SimulationParameters())

  # sensors
  displacement_sensor = robot_sensors.BaseDisplacementAndRotateSensor if rotate_sensor else robot_sensors.BaseDisplacementSensor
  sensors = [
    sensor_wrappers.HistoricSensorWrapper(
      wrapped_sensor=robot_sensors.MotorAngleSensor(
        num_motors=a1.NUM_MOTORS), num_history=3),
    sensor_wrappers.HistoricSensorWrapper(
      wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),]

  if not no_displacement:
    sensors.append(
      sensor_wrappers.HistoricSensorWrapper(
        wrapped_sensor=displacement_sensor(), num_history=3),)

  if goal:
    sensors.append(environment_sensors.GoalPosSensor())

  if add_last_action_input:
    sensors.append(sensor_wrappers.HistoricSensorWrapper(
      wrapped_sensor=environment_sensors.LastActionSensor(num_actions=a1.NUM_MOTORS), num_history=3))

  # terrain
  if terrain_type == "mount" or terrain_type == "hill":
    check_contact = True
  
  # task
  if goal:
    task = goal_task.GoalTask(
      z_constrain=z_constrain,
      other_direction_penalty=other_direction_penalty,
      z_penalty=z_penalty,
      num_action_repeat=num_action_repeat,
      time_step_s=time_step_s,
      height_fall_coeff=0.2,
      alive_reward=alive_reward,
      fall_reward=fall_reward,
      target_vel=target_vel,
      check_contact=check_contact,
      goal_coeff=goal_coeff,
      subgoal=subgoal
    )
  else:
    task = move_forward_task.MoveForwardTask(
      z_constrain=z_constrain,
      other_direction_penalty=other_direction_penalty,
      z_penalty=z_penalty,
      num_action_repeat=num_action_repeat,
      time_step_s=time_step_s,
      height_fall_coeff=0.2,
      alive_reward=alive_reward,
      fall_reward=fall_reward,
      target_vel=target_vel,
      check_contact=check_contact,
      subgoal_reward=subgoal_reward
    )

  # randomizer
  randomizers = []
  if domain_randomization:
    randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(
      verbose=False, fixed_delay_observation=fixed_delay_observation)
    randomizers.append(randomizer)
  terrain_randomizer = a1_rg.TerrainRandomizer(
    mesh_filename='terrain9735.obj',
    terrain_type=a1_rg.TerrainTypeDict[terrain_type],
    mesh_scale=[0.6, 0.3, 0.2],
    height_range=0.1,
    random_shape=random_shape,
    moving=moving
  )
  randomizers.append(terrain_randomizer)

  # init pose
  init_pos = None
  init_ori = None
  init_pos = a1_rg.QUADRUPED_INIT_POSITION[terrain_type]
  if "mount" in terrain_type:
    init_ori = a1_rg.QUADRUPED_INIT_ORI[terrain_type]

  # create gym env, NOTE: parameters are set here!
  env = locomotion_gym_env_with_rich_information.LocomotionGymEnv(
    gym_config=gym_config,
    robot_class=a1.A1,
    robot_sensors=sensors,
    env_randomizers=randomizers,
    get_image=get_image,
    depth_image=depth_image,
    grayscale=grayscale,
    rgbd=rgbd,
    depth_norm=depth_norm,
    fric_coeff=fric_coeff,
    task=task,
    random_init_range=random_init_range,
    init_pos=init_pos,
    init_ori=init_ori,
    frame_extract=frame_extract,
    record_video=record_video,
    get_image_interval=get_image_interval,
    reset_frame_idx=reset_frame_idx,
    reset_frame_idx_each_step=reset_frame_idx_each_step,
    interpolation=interpolation,
    fixed_delay_observation=fixed_delay_observation
  )

  # env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  if clip_num is not None:
    assert motor_control_mode == "POSITION"
    env = ActionRestrain(env, clip_num)

  if diagonal_act:
    env = DiagonalAction(env)

  if random_dir:
    assert terrain_type == "mount" or terrain_type == "hill"
    env = RandoDirWrapper(env, dir_update_interval=dir_update_interval)

  if curriculum:
    env = curriculum_wrapper_env.CurriculumWrapperEnv(env, episode_length_start=1000,
                                                      episode_length_end=2000,
                                                      curriculum_steps=10000000,
                                                      num_parallel_envs=8)

  return env


if __name__ == "__main__":
  env = build_a1_ground_env(
    motor_control_mode="POSITION",
    z_constrain=False,
    other_direction_penalty=0,
    z_penalty=1,
    clip_num=[0.05, 0.5, 0.5, 0.05, 0.5, 0.5, 0.05, 0.5, 0.5, 0.05, 0.5, 0.5],
    enable_rendering=True,
    diagonal_act=False,
    num_action_repeat=16,
    time_step_s=0.001,
    add_last_action_input=True,
    enable_action_interpolation=False,
    enable_action_filter=False,
    domain_randomization=True,
    get_image=True,
    depth_image=False,
    depth_norm=False,
    grayscale=True,
    rgbd=False,
    fric_coeff=[0.8, 0.1, 0.1],
    terrain_type="random_blocks_sparse",
    alive_reward=0.1,
    fall_reward=0,
    target_vel=1,
    random_init_range=0,
    dir_update_interval=None,
    check_contact=False,
    random_dir=False,
    rotate_sensor=False,
    frame_extract=1,
    goal=False,
    subgoal=False,
    goal_coeff=10,
    subgoal_reward=None,
    record_video=False,
    no_displacement=False,
    get_image_interval=1,
    reset_frame_idx=False,
    reset_frame_idx_each_step=False,
    random_shape=True,
    moving=False,
    curriculum=False,
    interpolation=False,
    fixed_delay_observation=False,
  )
  import time
  import pybullet as p
  from utils.record_video import create_video

  c_t = time.time()
  env.reset()
  images = []

  width = 256
  height = 256
  fov = 120
  aspect = width / height
  near = 0.02
  far = 10
  view_matrix = p.computeViewMatrix([1.75, 0, 1.0], [0, 0, 0], [0, 0, 1])
  projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

  for j in range(5000):
    obs, _, done, _ = env.step(env.action_space.sample())
    imgs = p.getCameraImage(width,
                        height,
                        view_matrix,
                        projection_matrix)
    _, _, img, _, _ = imgs
    images.append(img)
    if done:
      print("reset")
      env.reset()
  create_video(images)
  print(time.time() - c_t)
  print(env.count_t)
  print(10000 / (time.time() - c_t))
