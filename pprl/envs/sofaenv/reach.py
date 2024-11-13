from __future__ import annotations

from typing import Literal

import gymnasium as gym
from sofa_env.scenes.reach.reach_env import (
    ActionType,
    ObservationType,
    ReachEnv,
    RenderMode,
)

from . import add_env_wrappers


def build(
    max_episode_steps: int,
    render_mode: Literal["headless", "human"],
    action_type: Literal["discrete", "continuous"],
    observation_type: Literal["pointcloud", "rgb", "rgbd"],
    image_shape: list[int],
    frame_skip: int,
    time_step: float,
    discrete_action_magnitude: float,
    distance_to_target_threshold: float,
    reward_amount_dict: dict,
    create_scene_kwargs: dict,
    add_rendering_to_info: bool = True,
    **kwargs,
) -> gym.Env:
    assert len(image_shape) == 2
    image_shape = tuple(image_shape)  # type: ignore
    render_mode = RenderMode[render_mode.upper()]  # type: ignore
    action_type = ActionType[action_type.upper()]  # type: ignore

    if observation_type in ("pointcloud", "rgbd_image"):
        obs_type = ObservationType.RGBD
    elif observation_type == "rgb":
        obs_type = ObservationType.RGB
    else:
        raise ValueError(f"Invalid observation type: {observation_type}")

    env = ReachEnv(
        observation_type=obs_type,
        render_mode=render_mode,
        action_type=action_type,
        observe_target_position=False,
        image_shape=image_shape,
        frame_skip=frame_skip,
        time_step=time_step,
        discrete_action_magnitude=discrete_action_magnitude,
        distance_to_target_threshold=distance_to_target_threshold,
        reward_amount_dict=reward_amount_dict,
        create_scene_kwargs=create_scene_kwargs,
    )
    env = add_env_wrappers(
        env,
        max_episode_steps=max_episode_steps,
        add_rendering_to_info=add_rendering_to_info,
        observation_type=observation_type,
        **kwargs,
    )
    return env
