import argparse
from collections import defaultdict
from logging import critical, debug, error, info, warning
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D

import sys
sys.path.append('..')
from envs import create_env
from utils.tools import *

def main(env_id='MiniGrid-MazeS11N-v0',
         policy='random',
         num_steps=int(1e6),
         env_no_terminal=False,
         env_time_limit=0,
         env_action_repeat=1,
         ):

    # Env
    env = create_env(env_id, env_no_terminal, env_time_limit, env_action_repeat)

    # Policy
    policy = RandomPolicy(env.action_space)

    steps = 0
    while steps < num_steps:
        # Unroll one episode
        epsteps = 0
        obs = env.reset()
        done = False
        metrics = defaultdict(list)

        while not done:
            action, mets = policy(obs)
            obs, reward, done, inf = env.step(action)
            steps += 1
            epsteps += 1
            for k, v in mets.items():
                metrics[k].append(v)


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs) -> Tuple[int, dict]:
        return self.action_space.sample(), {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--num_steps', type=int, default=1_000_000)
    parser.add_argument('--env_time_limit', type=int, default=0)
    parser.add_argument('--env_action_repeat', type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))