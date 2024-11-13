import datetime
import logging
from pathlib import Path

import hydra
import numpy as np
import open3d as o3d
from gymnasium.utils.play import play
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from pprl.utils.o3d import np_to_o3d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("play_env")


@hydra.main(version_base=None, config_path="../conf", config_name="play_maniskill_env")
def main(config: DictConfig) -> None:
    with open_dict(config.env):
        env_name = config.env.pop("name")
        config.env.pop("traj_info")

    env_factory = instantiate(config.env, _convert_="partial", _partial_=True)
    seed = config.get("seed")

    env = env_factory()
    reset_obs, reset_info = env.reset(seed=seed)

    if (qpos := config.get("set_qpos")) is not None:
        qpos = np.array(qpos)
        env.unwrapped.agent.robot.set_qpos(qpos)
        logger.info(f"Setting qpos to {qpos}")

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    pcd = np_to_o3d(reset_obs["points"])
    visualizer.add_geometry(pcd)

    def render_pcd(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        pcd = np_to_o3d(obs_tp1["points"])
        visualizer.clear_geometries()
        visualizer.add_geometry(pcd)
        visualizer.poll_events()
        visualizer.update_renderer()

    play(
        env,
        keys_to_action={
            "q": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            "a": np.array([-1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            "w": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            "s": np.array([0, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            "e": np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
            "d": np.array([0, 0, -1, 0, 0, 0, 0, 0], dtype=np.float32),
            "r": np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
            "f": np.array([0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32),
            "t": np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            "g": np.array([0, 0, 0, 0, -1, 0, 0, 0], dtype=np.float32),
            "z": np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
            "h": np.array([0, 0, 0, 0, 0, -1, 0, 0], dtype=np.float32),
            "u": np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
            "j": np.array([0, 0, 0, 0, 0, 0, -1, 0], dtype=np.float32),
            "i": np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            "k": np.array([0, 0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
        },
        noop=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        callback=render_pcd,
    )

    visualizer.destroy_window()

    qpos = env.unwrapped.agent.robot.get_qpos()
    logger.info(f"qpos: {qpos}")

    model_id = env.unwrapped.model_id
    logger.info(f"Model id: {model_id}")

    if (save_dir := config.get("save_folder")) is not None:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = Path.home() / save_dir / f"{env_name}_{model_id}_{now}.ply"
        noop = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        obs, rew, terminated, truncated, info = env.step(noop)
        pcd = np_to_o3d(obs["points"])
        o3d.io.write_point_cloud(str(path), pcd)

    env.close()


if __name__ == "__main__":
    main()
