from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import gymnasium.spaces as spaces
import hydra
import parllel.logger as logger
import torch
import wandb
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from parllel import Array, ArrayDict, dict_map
from parllel.callbacks.recording_schedule import RecordingSchedule
from parllel.logger import Verbosity
from parllel.patterns import build_cages, build_sample_tree
from parllel.replays.replay import ReplayBuffer
from parllel.runners import RLRunner
from parllel.samplers import BasicSampler
from parllel.samplers.eval import EvalSampler
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.algos.sac import build_replay_buffer_tree
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.transforms.vectorized_video import RecordVectorizedVideo
from parllel.types import BatchSpec
import numpy as np

from pprl.utils.array_dict import build_obs_array

from omegaconf import OmegaConf

CAMERA_MISMATCHES = {
    "none": None,
    "roll15": {"roll": 15},
    "roll30": {"roll": 30},
    "x50": {"shift": [0.05, 0.0, 0.0]},
    "y50": {"shift": [0.0, 0.05, 0.0]},
    "z50": {"shift": [0.0, 0.0, 0.05]},
    "view50": {"view": 0.05},
}


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    v_q = np.array([v[0], v[1], v[2], 0.0])
    return quat_mul(quat_mul(q, v_q), q_conj)[:3]

"""
create_scene_kwargs:
  hole_config:
    inner_radius: 8.0
    outer_radius: 25.0
    height: 30.0
    young_modulus: 5000.0
    poisson_ratio: 0.3
    total_mass: 10.0
  thread_config:
    length: 70.0
    radius: 2.0
    total_mass: 1.0
    young_modulus: 1000.0
    poisson_ratio: 0.3
    beam_radius: 3.0
    mechanical_damping: 0.2
  gripper_config:
    cartesian_workspace:
      low: [-100.0, -100.0, 0.0]
      high: [100.0, 100.0, 200.0]
    state_reset_noise: [15.0, 15.0, 0.0, 20.0]
    rcm_reset_noise: null
    gripper_ptsd_state: [60.0, 0.0, 180.0, 90.0]
    gripper_rcm_pose: [100.0, 0.0, 150.0, 0.0, 180.0, 0.0]
  camera_config:
    placement_kwargs:
      position: [0.0, -175.0, 120.0]
      lookAt: [10.0, 0.0, 55.0]
    vertical_field_of_view: 62.0
"""


@contextmanager
def build(config: DictConfig) -> Iterator[RLRunner]:
    parallel = config.parallel
    discount = config.algo.discount
    batch_spec = BatchSpec(config.batch_T, config.batch_B)
    storage = "shared" if parallel else "local"

    """TRAIN CAGE"""

    with open_dict(config.env):
        config.env.pop("name")
        traj_info = config.env.pop("traj_info")

    TrajInfoClass = get_class(traj_info)
    TrajInfoClass.set_discount(discount)


    env_factory = instantiate(config.env, _convert_="partial", _partial_=True)

    cages, metadata = build_cages(
        EnvClass=env_factory,
        n_envs=batch_spec.B,
        TrajInfoClass=TrajInfoClass,
        parallel=parallel,
    )

    mismatch = CAMERA_MISMATCHES.get(config.camera_mismatch)
    if mismatch:
        camera_cfgs = config.env.env_kwargs.setdefault("camera_cfgs", {})
        pose = camera_cfgs.setdefault("pose", {"p": [0.0, 0.0, 0.0], "q": [0.0, 0.0, 0.0, 1.0]})
        p = np.array(pose.get("p", [0.0, 0.0, 0.0]), dtype=float)
        q = np.array(pose.get("q", [0.0, 0.0, 0.0, 1.0]), dtype=float)
        if "shift" in mismatch:
            p += np.array(mismatch["shift"], dtype=float)
        if "view" in mismatch:
            forward = rotate_vec(q, np.array([0.0, 0.0, 1.0]))
            p += forward * float(mismatch["view"])
        if "roll" in mismatch:
            angle = np.deg2rad(float(mismatch["roll"]))
            s, c = np.sin(angle / 2.0), np.cos(angle / 2.0)
            roll_q = np.array([0.0, 0.0, s, c])
            q = quat_mul(q, roll_q)
        pose["p"] = p.tolist()
        pose["q"] = q.tolist()

    env_factory = instantiate(config.env, _convert_="partial", _partial_=True)

    """EVAL CAGE"""

    eval_cages, eval_metadata = build_cages(
        EnvClass=env_factory,
        n_envs=config.eval.n_eval_envs,
        env_kwargs={"add_rendering_to_info": True},
        TrajInfoClass=TrajInfoClass,
        parallel=parallel,
    )

    # eval_cages, eval_metadata = build_cages(
    #     EnvClass=env_factory,
    #     n_envs=config.eval.n_eval_envs,
    #     env_kwargs={"add_rendering_to_info": True},
    #     TrajInfoClass=TrajInfoClass,
    #     parallel=parallel,
    # )

    replay_length = int(config.algo.replay_length) // batch_spec.B
    replay_length = (replay_length // batch_spec.T) * batch_spec.T
    sample_tree, metadata = build_sample_tree(
        env_metadata=metadata,
        batch_spec=batch_spec,
        parallel=parallel,
        full_size=replay_length,
        keys_to_skip=("obs", "next_obs"),
    )

    obs_space, action_space = metadata.obs_space, metadata.action_space

    sample_tree["observation"] = build_obs_array(
        metadata.example_obs,
        obs_space,
        batch_shape=tuple(batch_spec),
        storage=storage,
        padding=1,
        full_size=replay_length,
    )

    sample_tree["next_observation"] = sample_tree["observation"].new_array(
        padding=0,
        inherit_full_size=True,
    )

    assert isinstance(action_space, spaces.Box)
    n_actions = action_space.shape[0]

    # create model
    model = torch.nn.ModuleDict()

    with open_dict(config.model):
        encoder_name = config.model.pop("name")

    if encoder_name != "Passthru":
        encoder = instantiate(
            config.model,
            _convert_="partial",
            obs_space=obs_space,
        )
        model["encoder"] = encoder
        embedding_size = encoder.embed_dim
    else:
        embedding_size = spaces.flatdim(obs_space)

    model["pi"] = instantiate(
        config.pi_mlp_head,
        input_size=embedding_size,
        action_size=n_actions,
        action_space=action_space,
        _convert_="partial",
    )
    model["q1"] = instantiate(
        config.q_mlp_head,
        input_size=embedding_size,
        action_size=n_actions,
        _convert_="partial",
    )
    model["q2"] = instantiate(
        config.q_mlp_head,
        input_size=embedding_size,
        action_size=n_actions,
        _convert_="partial",
    )

    distribution = SquashedGaussian(
        dim=n_actions,
        scale=action_space.high[0],
    )
    device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
    device = torch.device(device)

    # instantiate agent
    agent = SacAgent(
        model=model,
        distribution=distribution,
        device=device,
        learning_starts=config.algo.learning_starts,
    )

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_tree=sample_tree,
    )

    # create replay buffer
    replay_buffer_tree = build_replay_buffer_tree(sample_tree)

    def batch_transform(tree: ArrayDict[Array]) -> ArrayDict[torch.Tensor]:
        tree = tree.to_ndarray()  # type: ignore
        tree = tree.apply(torch.from_numpy)
        return tree.to(device=device)

    replay_buffer = ReplayBuffer(
        tree=replay_buffer_tree,
        sampler_batch_spec=batch_spec,
        size_T=replay_length,
        replay_batch_size=config.algo.batch_size,
        newest_n_samples_invalid=0,
        oldest_n_samples_invalid=1,
        batch_transform=batch_transform,
    )

    # create optimizers
    with open_dict(config.optimizer):
        q_optim_conf = config.optimizer.pop("q", {}) or {}
        pi_optim_conf = config.optimizer.pop("pi", {}) or {}
        encoder_optim_conf = config.optimizer.pop("encoder", {}) or {}

    pi_optimizer = instantiate(
        config.optimizer,
        [{"params": agent.model["pi"].parameters(), **pi_optim_conf}],
    )

    q_optimizer = instantiate(
        config.optimizer,
        [
            {"params": agent.model["q1"].parameters(), **q_optim_conf},
            {"params": agent.model["q2"].parameters(), **q_optim_conf},
        ],
    )
    if "encoder" in agent.model:
        q_optimizer.add_param_group(
            {"params": agent.model["encoder"].parameters(), **encoder_optim_conf}
        )

    # create learning rate schedulers
    if gamma := config.get("lr_scheduler_gamma") is not None:
        pi_scheduler = torch.optim.lr_scheduler.ExponentialLR(pi_optimizer, gamma=gamma)
        q_scheduler = torch.optim.lr_scheduler.ExponentialLR(q_optimizer, gamma=gamma)
        lr_schedulers = [pi_scheduler, q_scheduler]
    else:
        lr_schedulers = None

    # create algorithm
    algorithm = instantiate(
        config.algo,
        _convert_="partial",
        batch_spec=batch_spec,
        agent=agent,
        replay_buffer=replay_buffer,
        q_optimizer=q_optimizer,
        pi_optimizer=pi_optimizer,
        learning_rate_schedulers=lr_schedulers,
        action_space=action_space,
    )

    logger.debug("Allocating eval sample tree...")

    eval_tree_keys = [
        "action",
        "agent_info",
        "observation",
        "reward",
        "terminated",
        "truncated",
        "done",
    ]
    eval_tree_example = ArrayDict({key: sample_tree[key] for key in eval_tree_keys})
    # create a new tree with leading dimensions (1, B_eval)
    eval_sample_tree = eval_tree_example.new_array(
        batch_shape=(1, config.eval.n_eval_envs)
    )

    # add env_info according to template from eval envs, since the env_info is different
    eval_sample_tree["env_info"] = dict_map(
        Array.from_numpy,
        eval_metadata.example_info,
        batch_shape=(1, config.eval.n_eval_envs),
        storage=storage,
    )

    video_recorder = RecordVectorizedVideo(
        sample_tree=eval_sample_tree,
        buffer_key_to_record="env_info.rendering",
        env_fps=50,
        output_dir=Path(config.video_path),
        video_length=config.env.max_episode_steps,
        use_wandb=True,
    )
    recording_schedule = RecordingSchedule(video_recorder, trigger="on_eval")

    eval_sampler = EvalSampler(
        max_traj_length=config.env.max_episode_steps,
        max_trajectories=config.eval.max_trajectories,
        envs=eval_cages,
        agent=agent,
        sample_tree=eval_sample_tree,
        step_transforms=[video_recorder],
    )

    # create runner
    runner = RLRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        eval_sampler=eval_sampler,
        n_steps=config.runner.n_steps,
        log_interval_steps=config.runner.log_interval_steps,
        eval_interval_steps=config.runner.eval_interval_steps,
        callbacks=[recording_schedule],
    )

    try:
        yield runner

    finally:
        eval_sampler.close()
        eval_sample_tree.close()
        sampler.close()
        sample_tree.close()
        agent.close()
        for eval_cage in eval_cages:
            eval_cage.close()
        for cage in cages:
            cage.close()


@hydra.main(version_base=None, config_path="../conf", config_name="train_sac")
def main(config: DictConfig) -> None:
    with open_dict(config):
        wandb_config = config.pop("wandb", {})
        notes = wandb_config.pop("notes", None)
        tags = wandb_config.pop("tags", None)
        group_name = wandb_config.pop("group_name", None)
        print("Group name is: ", group_name)

    run = wandb.init(
        project="pprl",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),  # type: ignore
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
        reinit=True,  # required for hydra sweeps with default launcher
        tags=tags,
        notes=notes,
        group=group_name,
    )

    logger.init(
        wandb_run=run,
        # this log_dir is used if wandb is disabled (using `wandb disabled`)
        log_dir=Path(f"log_data/sac/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        tensorboard=True,
        output_files={
            "txt": "log.txt",
            # "csv": "progress.csv",
        },  # type: ignore
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),  # type: ignore
        model_save_path=Path("model.pt"),
        # verbosity=Verbosity.DEBUG,
    )

    video_path = (
        Path(config.video_path)
        / f"{datetime.now().strftime('%Y-%m-%d')}/{run.id}"  # type: ignore
    )
    config.update({"video_path": video_path})

    model_path = Path(config.get("model_path", "model.pt"))
    with build(config) as runner:
        runner.run()
        torch.save(runner.agent.state_dict(), model_path)

    logger.close()
    run.finish()  # type: ignore


if __name__ == "__main__":
    main()
