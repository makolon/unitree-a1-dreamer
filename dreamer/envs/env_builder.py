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

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from envs import locomotion_gym_env
from envs import locomotion_gym_env_with_rich_information
from envs import locomotion_gym_config
import envs.utilities.a1_random_ground as a1_rg
from envs.env_wrappers import imitation_wrapper_env
from envs.env_wrappers import observation_dictionary_to_array_wrapper
from envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from envs.env_wrappers import trajectory_generator_wrapper_env
from envs.env_wrappers import goal_task
from envs.env_wrappers import move_forward_task
from envs.env_wrappers import simple_openloop
from envs.env_wrappers import simple_forward_task
from envs.env_wrappers import imitation_task
from envs.env_wrappers import default_task

from envs.sensors import environment_sensors
from envs.sensors import sensor_wrappers
from envs.sensors import robot_sensors
from envs.utilities import controllable_env_randomizer_from_config
from robots import laikago
from robots import a1
from robots import robot_config


def build_laikago_env( motor_control_mode, enable_rendering):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.motor_control_mode = motor_control_mode
  sim_params.reset_time = 2
  sim_params.num_action_repeat = 10
  sim_params.enable_action_interpolation = False
  sim_params.enable_action_filter = False
  sim_params.enable_clip_motor_commands = False
  
  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  robot_class = laikago.Laikago

  sensors = [
      robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS),
      robot_sensors.IMUSensor(),
      environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS)
  ]

  task = default_task.DefaultTask()

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                            robot_sensors=sensors, task=task)

  #env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  #env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
  #                                                                     trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=laikago.UPPER_BOUND))

  return env


def build_imitation_env(motion_files, num_parallel_envs, mode,
                        enable_randomizer, enable_rendering,
                        robot_class=laikago.Laikago,
                        trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=laikago.UPPER_BOUND)):
  assert len(motion_files) > 0

  curriculum_episode_length_start = 20
  curriculum_episode_length_end = 600
  
  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.allow_knee_contact = True
  sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION

  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  sensors = [
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS), num_history=3)
  ]

  task = imitation_task.ImitationTask(ref_motion_filenames=motion_files,
                                      enable_cycle_sync=True,
                                      tar_frame_steps=[1, 2, 10, 30],
                                      ref_state_init_prob=0.9,
                                      warmup_time=0.25)

  randomizers = []
  if enable_randomizer:
    randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
    randomizers.append(randomizer)

  env = locomotion_gym_env_with_rich_information.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                            env_randomizers=randomizers, robot_sensors=sensors, task=task)

  env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
                                                                       trajectory_generator=trajectory_generator)

  if mode == "test":
      curriculum_episode_length_start = curriculum_episode_length_end

  env = imitation_wrapper_env.ImitationWrapperEnv(env,
                                                  episode_length_start=curriculum_episode_length_start,
                                                  episode_length_end=curriculum_episode_length_end,
                                                  curriculum_steps=30000000,
                                                  num_parallel_envs=num_parallel_envs)
  return env



def build_regular_env(
    robot_class=a1.A1,
    motor_control_mode=robot_config.MotorControlMode.POSITION,
    z_constrain=False,
    other_direction_penalty=0,
    z_penalty=0,
    clip_num=None,
    enable_rendering=False,
    diagonal_act=False,
    num_action_repeat=10,
    time_step_s=0.001,
    add_last_action_input=False,
    enable_action_interpolation=False,
    enable_action_filter=False,
    domain_randomization=False,
    get_image=True,
    depth_image=False,
    depth_norm=False,
    grayscale=True,
    rgbd=False,
    fric_coeff=[0.8, 0.1, 0.1],
    terrain_type="plane",
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
    random_shape=False,
    moving=False,
    curriculum=False,
    interpolation=False,
    fixed_delay_observation=False,
    action_limit=(0.75, 0.75, 0.75),
    wrap_trajectory_generator=True
):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.motor_control_mode = motor_control_mode
  sim_params.reset_time = 2
  sim_params.num_action_repeat = 10
  sim_params.enable_action_interpolation = False
  sim_params.enable_action_filter = False
  sim_params.enable_clip_motor_commands = False
  sim_params.robot_on_rack = False

  gym_config = locomotion_gym_config.LocomotionGymConfig(
      simulation_parameters=sim_params)

  sensors = [
      robot_sensors.BaseDisplacementSensor(),
      robot_sensors.IMUSensor(),
      robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS),
  ]

  if terrain_type == "mount" or terrain_type == "hill":
    check_contact = True
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

  init_pos, init_ori = None, None
  env = locomotion_gym_env_with_rich_information.LocomotionGymEnv(
    gym_config=gym_config,
    robot_class=robot_class,
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

  # env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  if (motor_control_mode == robot_config.MotorControlMode.POSITION) and wrap_trajectory_generator:
    if robot_class == laikago.Laikago:
      env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
          trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=action_limit))
    elif robot_class == a1.A1:
      env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
          trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=action_limit))
  return env