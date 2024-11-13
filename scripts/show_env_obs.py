import logging

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


@hydra.main(version_base=None, config_path="../conf", config_name="show_env_obs")
def main(config: DictConfig) -> None:
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

    isolate = config.get("isolate")
    isolate = isolate if isolate is not None else default_isolate

    seed = config.get("seed")

    if isolate:
        # isolate env process from main process to prevent interference with
        # rendering

        env = ProcessCage(EnvClass=env_factory, seed=seed)
        obs_space = env.spaces.observation

        env.random_step_async()
        action, next_obs, obs, reward, terminated, truncated, info = env.await_step()
        env.close()

    else:
        env = env_factory()
        env.reset(seed=seed)
        obs_space = env.observation_space

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()

    if env_suite == "maniskill2":
        if isinstance(obs_space, PointCloudSpace):
            print(f"Observation has {len(obs)} points")
            pcd = np_to_o3d(obs)
            o3d.visualization.draw_geometries([pcd])

        elif isinstance(obs_space, spaces.Box):
            import matplotlib.pyplot as plt

            image = np.moveaxis(obs, 0, -1)

            plt.imshow(image)
            plt.show()

        elif isinstance(obs_space, spaces.Dict):
            if "points" in obs:
                print(f"Observation has {len(obs['points'])} points")
                pcd = np_to_o3d(obs["points"])
                o3d.visualization.draw_geometries([pcd])

            if "image" in obs:
                import matplotlib.pyplot as plt

                for name, camera in obs["image"].items():
                    plt.imshow(camera["Color"][..., :3])
                    plt.title(name)
                    plt.show()
        else:
            raise NotImplementedError(obs_space)

    elif env_suite == "sofaenv":
        if isinstance(obs_space, PointCloudSpace):
            print(f"Observation has {len(obs)} points")
            pcd = np_to_o3d(obs)
            o3d.visualization.draw_geometries([pcd])
        else:
            raise NotImplementedError(obs_space)


if __name__ == "__main__":
    main()
