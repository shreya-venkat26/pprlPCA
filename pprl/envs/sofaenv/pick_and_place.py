from __future__ import annotations

from collections import deque
from typing import Any, Literal

import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from sofa_env.base import RenderMode
from sofa_env.scenes.pick_and_place.pick_and_place_env import (
    ActionType,
    ObservationType,
    Phase,
)
from sofa_env.scenes.pick_and_place.pick_and_place_env import (
    PickAndPlaceEnv as _PickAndPlaceEnv,
)

from . import SofaAddRenderingToInfoWrapper
from .pointcloud_obs import SofaEnvPointCloudObservations


def build(
    max_episode_steps: int,
    render_mode: Literal["headless", "human"],
    action_type: Literal["discrete", "continuous"],
    image_shape: list[int],
    frame_skip: int,
    time_step: float,
    phase_any_rewards: dict,
    phase_pick_rewards: dict,
    phase_place_rewards: dict,
    randomize_torus_position: bool,
    create_scene_kwargs: dict = {},
    add_rendering_to_info: bool = False,
):
    assert len(image_shape) == 2
    image_shape = tuple(image_shape)  # type: ignore
    render_mode = RenderMode[render_mode.upper()]  # type: ignore
    action_type = ActionType[action_type.upper()]  # type: ignore

    env = PickAndPlaceEnv(
        observation_type=ObservationType.RGBD,
        render_mode=render_mode,
        action_type=action_type,
        image_shape=image_shape,
        frame_skip=frame_skip,
        time_step=time_step,
        randomize_torus_position=randomize_torus_position,
        reward_amount_dict={
            Phase.ANY: phase_any_rewards,
            Phase.PICK: phase_pick_rewards,
            Phase.PLACE: phase_place_rewards,
        },
        create_scene_kwargs=create_scene_kwargs,
    )
    if add_rendering_to_info:
        env = SofaAddRenderingToInfoWrapper(env)

    env = SofaEnvPointCloudObservations(env)
    env = TimeLimit(env, max_episode_steps)
    return env


class PickAndPlaceEnv(_PickAndPlaceEnv):
    def reset(
        self,
        seed: int | np.random.SeedSequence | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | None, dict]:
        """Reset the state of the environment and return the initial observation."""

        if self._initialized:
            # If the gripper does not start already having grasped the torus, we can randomize the torus' starting position.
            # This has to happen before resetting the SOFA simulation, because setting the state of the MO in Rope
            # currently does not work -> we have to change the reset position -> SOFA will place the torus on reset.
            if self.randomize_torus_position:
                # Get a new position randomly from the board
                new_center_of_mass = self.rng.uniform([-80, 40, -80], [80, 40, 80])
                # Translation to the new position from the old one
                translation_offset = (
                    new_center_of_mass - self.torus.get_reset_center_of_mass()
                )
                new_states = self.torus.get_reset_state().copy()
                new_states[:, :3] += translation_offset
                self.torus.set_reset_state(new_states)

            new_position = self.rng.uniform([-80, 0, -80], [80, 0, 80])
            translation_offset = (
                new_position - self.pegs[0].mechanical_object.position.array()[:, :3]
            )
            new_state = self.pegs[0].mechanical_object.reset_position.array().copy()
            new_state[:, :3] += translation_offset
            with self.pegs[0].mechanical_object.position.writeable() as sofa_state:
                sofa_state[:] = new_state

        # Reset from parent class -> calls the simulation's reset function
        super(_PickAndPlaceEnv, self).reset(seed)

        # Seed the instruments
        if self.unconsumed_seed:
            seeds = self.seed_sequence.spawn(1)
            self.gripper.seed(seed=seeds[0])
            self.unconsumed_seed = False

        # Reset the gripper
        self.gripper.reset_gripper()
        self.previous_gripper_position[:] = self.gripper.get_pose()[:3]

        # Chose new active pegs
        active_indices = self.rng.choice(
            len(self.pegs), size=self.num_active_pegs, replace=False
        )
        self.active_target_positions[:] = self.target_positions[active_indices]
        self.torus_distance_to_active_pegs = np.linalg.norm(
            self.torus.get_center_of_mass() - self.active_target_positions, axis=1
        )

        # # Randomize colors if required
        # if self.randomize_color:
        #     active_color_index, inactive_color_index = self.rng.choice(len(self.colors) - 1, size=2, replace=False)
        #     active_color = self.colors[active_color_index]
        #     inactive_color = self.colors[inactive_color_index]
        # else:
        #     active_color = (255, 0, 0)
        #     inactive_color = (0, 0, 255)

        # # Set colors of the pegs and torus
        # for peg in self.pegs:
        #     peg.set_color(inactive_color)
        # for index in active_indices:
        #     self.pegs[index].set_color(active_color)
        # self.torus.set_color(active_color)

        # Clear the episode info values
        for key in self.episode_info:
            self.episode_info[key] = 0.0

        # and the reward info dict
        self.reward_info = {}

        # and fill the first values used as previous_reward_features
        self.reward_features = {}
        self.reward_features["grasped_torus"] = self.gripper.grasp_established
        self.reward_features["torus_distance_to_active_pegs"] = np.linalg.norm(
            self.torus.get_center_of_mass() - self.active_target_positions, axis=1
        )
        self.reward_features["gripper_distance_to_torus_center"] = np.linalg.norm(
            self.gripper.get_grasp_center_position() - self.torus.get_center_of_mass()
        )
        if self.torus_tracking_point_indices is not None:
            self.reward_features[
                "gripper_distance_to_torus_tracking_points"
            ] = np.linalg.norm(
                self.gripper.get_grasp_center_position()
                - self.torus.get_positions()[self.torus_tracking_point_indices],
                axis=1,
            )

        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(
                self._sofa_root_node, self._sofa_root_node.getDt()
            )

        # If the torus is immediately unstable after reset, do reset again
        # The torus is probably unstable, if the mean velocity is larger than 25 mm/s
        mean_torus_velocity = np.mean(
            np.linalg.norm(self.torus.mechanical_object.velocity.array()[:, :3], axis=1)
        )
        torus_probably_unstable = mean_torus_velocity > 25.0
        if torus_probably_unstable:
            print("Reset again, because simulation was unstable!")
            self.reset()

        # Set the current phase after reset
        self.active_phase = Phase.PLACE if self.start_grasped else Phase.PICK

        return (
            self._get_observation(
                maybe_rgb_observation=self._maybe_update_rgb_buffer()
            ),
            {},
        )
