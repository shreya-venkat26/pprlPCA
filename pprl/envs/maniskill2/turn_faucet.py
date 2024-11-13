from __future__ import annotations

from typing import Sequence

from mani_skill2 import format_path
from mani_skill2.envs.misc.turn_faucet import TurnFaucetEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env


@register_env("TurnFaucet-v1", max_episode_steps=200)
class ConfigurableTurnFaucetEnv(TurnFaucetEnv):
    """This version of the environment provides more options for camera
    configuration and the number of faucet models used.
    """

    def __init__(
        self,
        *args,
        observe_render_cam: bool = False,
        robot_cameras: Sequence[str] | None = None,
        n_models: int | None = None,
        **kwargs,
    ) -> None:
        if n_models is not None:
            model_json = format_path(
                "{PACKAGE_ASSET_DIR}/partnet_mobility/meta/info_faucet_train.json"
            )
            model_db: dict[str, dict] = load_json(model_json)
            kwargs["model_ids"] = list(model_db.keys())[:n_models]

        self.observe_render_cam = observe_render_cam
        self.robot_cameras = robot_cameras
        super().__init__(*args, **kwargs)

    def _register_cameras(self) -> list[CameraConfig]:
        # there should be one existing camera (cameras mounted on the robot
        # are not included)
        cameras = super()._register_cameras()
        assert isinstance(cameras, CameraConfig)
        cameras = [cameras]

        if self.observe_render_cam:
            # get the config also used for the render camera
            render_cam = self._register_render_cameras()
            cameras += [render_cam]

        return cameras

    def _configure_cameras(self) -> None:
        super()._configure_cameras()
        if self.robot_cameras is not None:
            self._camera_cfgs = {
                uid: cfg
                for uid, cfg in self._camera_cfgs.items()
                if uid in self.robot_cameras
            }
