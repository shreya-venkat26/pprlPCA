from __future__ import annotations

import gymnasium as gym
import mani_skill2.envs  # to register maniskill envs

# to register modified envs
import pprl.envs.maniskill2.open_cabinet_door_drawer
import pprl.envs.maniskill2.pick_cube
import pprl.envs.maniskill2.push_chair
import pprl.envs.maniskill2.turn_faucet


class ManiSkillAddObsToInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, key: str = "rendering"):
        super().__init__(env)
        self.key = key

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info[self.key] = self.env.unwrapped.render_cameras()  # type: ignore
        return observation, reward, terminated, truncated, info


from .build import build

__all__ = ["build"]
