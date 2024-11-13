from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

from pprl.envs.wrappers import TransposeImageWrapper

from .pointcloud_obs import SofaEnvPointCloudObservations


class SofaAddRenderingToInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, key: str = "rendering"):
        super().__init__(env)
        self.key = key

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info[self.key] = observation[..., :3]
        return observation, reward, terminated, truncated, info


def add_env_wrappers(
    env: gym.Env,
    max_episode_steps: int,
    add_rendering_to_info: bool,
    observation_type: Literal["pointcloud", "rgb", "rgbd"],
    **kwargs,
) -> gym.Env:
    if add_rendering_to_info:
        env = SofaAddRenderingToInfoWrapper(env)

    if observation_type in ("rgb", "rgbd"):
        env = TransposeImageWrapper(env)
    elif observation_type in ("pointcloud"):
        env = SofaEnvPointCloudObservations(
            env,
            **kwargs,
        )
    else:
        raise NotImplementedError(observation_type)

    env = TimeLimit(env, max_episode_steps)
    return env


def convert_to_array(kwargs_dict):
    for k, v in kwargs_dict.items():
        if isinstance(v, list):
            kwargs_dict[k] = np.asarray(v)
        elif isinstance(v, dict):
            convert_to_array(v)
