import numpy as np
import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig, open_dict

from train_sac import build, CAMERA_MISMATCHES

MISMATCHES = [k for k in CAMERA_MISMATCHES if k != "none"]


@hydra.main(version_base=None, config_path="../conf", config_name="train_sac")
def main(config: DictConfig) -> None:
    state_dict = torch.load(Path(config.model_path), map_location=config.device)
    results = {}
    for name in MISMATCHES:
        with open_dict(config):
            config.camera_mismatch = name
        with build(config) as runner:
            runner.agent.load_state_dict(state_dict)
            stats = runner.eval_sampler.evaluate(n_trajs=config.eval.max_trajectories)
            successes = [info.get("success", 0.0) for info in stats]
            results[name] = float(np.mean(successes))
    print(results)


if __name__ == "__main__":
    main()
