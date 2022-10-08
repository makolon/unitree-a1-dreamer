"""Wrapper to make the a1 environment suitable for OpenAI gym."""
import gym

from robots import a1
from robots import robot_config
from envs import env_builder

class A1GymEnv(gym.Env):
  """A1 environment that supports the gym interface."""
  metadata = {'render.modes': ['rgb_array']}

  def __init__(self,
               action_limit=0.75,
               render=True,
               on_rack=False):
    self._env = env_builder.build_regular_env(
        robot_class=a1.A1,
        motor_control_mode=robot_config.MotorControlMode.POSITION,
        z_constrain=False,
        other_direction_penalty=0,
        z_penalty=0,
        clip_num=None,
        enable_rendering=render,
        diagonal_act=False,
        num_action_repeat=100,
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
        alive_reward=0.5,
        fall_reward=-10,
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
        action_limit=action_limit,
        wrap_trajectory_generator=True
    )
    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space

  def step(self, action):
    return self._env.step(action)

  def reset(self):
    return self._env.reset()

  def close(self):
    self._env.close()

  def render(self, mode):
    return self._env.render(mode)

  def __getattr__(self, attr):
    return getattr(self._env, attr)
