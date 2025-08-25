import numpy as np
import torch
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from train_sac import build

MISMATCHES = ["roll15", "roll30", "x50", "y50", "z50", "view50"]

@hydra.main(version_base=None, config_path="../conf", config_name="train_sac")
def main(config: DictConfig) -> None:
    state_dict = torch.load(Path(config.model_path), map_location=config.device)
    results = {}
    for name in MISMATCHES:
        mismatch_cfg = OmegaConf.load(Path("../conf/camera_mismatch") / f"{name}.yaml")
        with open_dict(config):
            config.eval_camera_mismatch = mismatch_cfg
        with build(config) as runner:
            runner.agent.load_state_dict(state_dict)
            stats = runner.eval_sampler.evaluate(n_trajs=config.eval.max_trajectories)
            successes = [info.get("success", 0.0) for info in stats]
            results[name] = float(np.mean(successes))
    print(results)

if __name__ == "__main__":
    main()
