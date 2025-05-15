import logging
import time
from pathlib import Path

import gymnasium.spaces as spaces
import hydra
import numpy as np
import open3d as o3d
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from parllel.cages import ProcessCage

from pprl.envs.pointcloud_space import PointCloudSpace
from pprl.utils.o3d import np_to_o3d

logging.basicConfig(level=logging.INFO)

# ---------- helper ----------------------------------------------------------
def save_point_cloud(points: np.ndarray, outfile: Path | str = "obs_pointcloud.ply") -> None:
    """
    Convert an (N,3)|(N,6)|(N,9) numpy point cloud to Open3D and save as PLY.
    """
    pcd = np_to_o3d(points)
    outfile = Path(outfile).expanduser().resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(outfile), pcd, write_ascii=False, compressed=True)
    logging.info(f"Point cloud with {len(points)} points written to {outfile}")


@hydra.main(version_base=None, config_path="../conf", config_name="show_env_obs")
def main(config: DictConfig) -> None:
    # ----------------------------------------------------------------------- #
    # 1. build env exactly as before                                           #
    # ----------------------------------------------------------------------- #
    with open_dict(config.env):
        env_name = config.env.pop("name")
        config.env.pop("traj_info")

    env_factory = instantiate(config.env, _convert_="partial", _partial_=True)

    if env_name in ("TurnFaucet", "OpenCabinetDrawer", "OpenCabinetDoor", "PushChair"):
        env_suite = "maniskill2"
        default_isolate = False
    elif env_name in ("ThreadInHole", "DeflectSpheres"):
        env_suite = "sofaenv"
        default_isolate = True
    else:
        raise ValueError(f"Unknown env name {env_name}")

    isolate = config.get("isolate")
    isolate = isolate if isolate is not None else default_isolate
    seed = config.get("seed")

    if isolate:
        env = ProcessCage(EnvClass=env_factory, seed=seed)
        env.random_step_async()
        action, next_obs, obs, reward, terminated, truncated, info = env.await_step()
        obs_space = env.spaces.observation
        env.close()
    else:
        env = env_factory()
        env.reset(seed=seed)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        obs_space = env.observation_space
        env.close()

    # ----------------------------------------------------------------------- #
    # 2. save the point cloud instead of opening a window                     #
    # ----------------------------------------------------------------------- #
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outfile   = f"{env_name}_obs_{timestamp}.ply"

    if env_suite in ("maniskill2", "sofaenv"):
        if isinstance(obs_space, PointCloudSpace):
            print(f"Observation has {len(obs)} points")
            save_point_cloud(obs, outfile)

        elif isinstance(obs_space, spaces.Dict) and "points" in obs:
            print(f"Observation has {len(obs['points'])} points")
            save_point_cloud(obs["points"], outfile)

        elif isinstance(obs_space, spaces.Box) or (
            isinstance(obs_space, spaces.Dict) and "image" in obs
        ):
            logging.warning("Current observation is an image — nothing to save.")
        else:
            raise NotImplementedError(obs_space)

if __name__ == "__main__":
    main()

