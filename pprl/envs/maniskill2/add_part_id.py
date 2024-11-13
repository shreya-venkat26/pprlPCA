from __future__ import annotations

from typing import Any

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from gymnasium import Env


class AddPartIdWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: Env,
        id_key: str = "part_id",
    ):
        super().__init__(env)

        self.n_model_ids = len(self.env.unwrapped.model_ids)
        lows = np.zeros((self.n_model_ids,), dtype=np.uint8)
        highs = np.ones((self.n_model_ids,), dtype=np.uint8)
        wrapped_space = self.env.observation_space

        if isinstance(wrapped_space, spaces.Box):
            assert len(wrapped_space.shape) == 1
            self.dict_observation = False
            lows = np.concatenate((wrapped_space.low, lows))
            highs = np.concatenate((wrapped_space.high, highs))
            self.dtype = lows.dtype
            self.observation_space = spaces.Box(low=lows, high=highs, dtype=self.dtype)

        elif isinstance(wrapped_space, spaces.Dict):
            self.dict_observation = True
            self.id_key = id_key
            self.dtype = np.uint8
            self.observation_space[id_key] = spaces.Box(
                low=lows, high=highs, dtype=self.dtype
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        model_name = self.env.unwrapped.model_id
        model_id = self.env.unwrapped.model_ids.index(model_name)
        self.model_id_encoding = np.zeros((self.n_model_ids,), dtype=self.dtype)
        self.model_id_encoding[model_id] = 1

        return self.observation(obs), info

    def observation(self, observation: Any) -> Any:
        if self.dict_observation:
            observation[self.id_key] = self.model_id_encoding.copy()
            return observation
        else:
            return np.concatenate((observation, self.model_id_encoding))
