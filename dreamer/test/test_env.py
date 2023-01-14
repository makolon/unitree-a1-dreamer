import argparse
from collections import defaultdict
from logging import critical, debug, error, info, warning
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
import pybullet as p
import sys
sys.path.append('..')
from envs import create_env
from utils.tools import *
import matplotlib.pyplot as plt
from utils.record_video import create_video

def main(env_id='MiniGrid-MazeS11N-v0',
         policy='random',
         num_steps=int(1e6),
         env_no_terminal=False,
         env_time_limit=0,
         env_action_repeat=16,
         ):

    # Env
    env = create_env(env_id, env_no_terminal, env_time_limit, env_action_repeat)
    steps = 0
    _ = env.reset()

    # Policy

    # Camera
    width = 1024
    height = 1204
    fov = 120
    aspect = width / height
    near = 0.02
    far = 10
    view_matrix = p.computeViewMatrix([1.75, 0, 1.0], [0, 0, 0], [0, 0, 1])
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    images = []
    while True:
        start_time = time.time()
        action = env.action_space.sample()
        obs, reward, done, inf = env.step(action)
        imgs = p.getCameraImage(width,
                            height,
                            view_matrix,
                            projection_matrix)
        _, _, img, _, _ = imgs
        img = obs['image']
        images.append(img)
        steps += 1
        print('steps: ', steps)
        print('fps: ', 1.0 / (time.time() - start_time))
        if steps > 1000:
            break
    create_video(images, 64, 64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--num_steps', type=int, default=1_000_000)
    parser.add_argument('--env_time_limit', type=int, default=0)
    parser.add_argument('--env_action_repeat', type=int, default=10)
    args = parser.parse_args()
    main(**vars(args))