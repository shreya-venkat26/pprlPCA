from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


class AddCabinetHandleMask(gym.Wrapper):
    def __init__(self, env: gym.Env, target_handle_only: bool = False):
        super().__init__(env)
        self.target_handle_only = target_handle_only

    def step(self, action) -> tuple[dict, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        observation, info = self.observation(observation, info)
        return observation, reward, terminated, truncated, info

    def observation(self, observation: dict, info: dict) -> tuple[dict, dict]:
        mesh_segmentation = observation["pointcloud"]["Segmentation"][..., 0]
        handle_mask = np.isin(mesh_segmentation, self.handle_mesh_ids)
        info["cabinet_handle"] = handle_mask
        return observation, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)

        if self.target_handle_only:
            target_link = self.env.unwrapped.target_links[
                self.env.unwrapped.target_link_idx
            ]
            self.handle_mesh_ids = [
                mesh.visual_id
                for mesh in target_link.get_visual_bodies()
                if "handle" in mesh.name
            ]
        else:
            cabinet_links = self.env.unwrapped.cabinet.get_links()
            self.handle_mesh_ids = [
                mesh.visual_id
                for link in cabinet_links
                for mesh in link.get_visual_bodies()
                if "handle" in mesh.name
            ]

        observation, info = self.observation(observation, info)
        return observation, info
