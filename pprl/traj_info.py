from dataclasses import dataclass

from parllel.cages.traj_info import (
    ActionType,
    DoneType,
    EnvInfoType,
    ObsType,
    RewardType,
    TrajInfo,
)


@dataclass
class SofaTrajInfo(TrajInfo):
    Success: bool = False

    def step(
        self,
        observation: ObsType,
        action: ActionType,
        reward: RewardType,
        terminated: DoneType,
        truncated: DoneType,
        env_info: EnvInfoType,
    ) -> None:
        super().step(observation, action, reward, terminated, truncated, env_info)
        self.Success = env_info["successful_task"]


@dataclass
class ManiTrajInfo(TrajInfo):
    Success: bool = False
    SuccessLength: int = 0

    def step(
        self,
        observation: ObsType,
        action: ActionType,
        reward: RewardType,
        terminated: DoneType,
        truncated: DoneType,
        env_info: EnvInfoType,
    ) -> None:
        super().step(observation, action, reward, terminated, truncated, env_info)
        self.Success = env_info["success"] or self.Success
        if not self.Success:
            self.SuccessLength += 1


@dataclass
class OpenCabinetDrawerTrajInfo(ManiTrajInfo):
    StageReturn: float = 0
    EECloseToHandle: float = 0
    OpenEnough: float = 0

    def step(
        self,
        observation: ObsType,
        action: ActionType,
        reward: RewardType,
        terminated: DoneType,
        truncated: DoneType,
        env_info: EnvInfoType,
    ) -> None:
        super().step(observation, action, reward, terminated, truncated, env_info)
        self.StageReturn += env_info["stage_reward"]
        self.EECloseToHandle += float(env_info["ee_close_to_handle"])
        self.OpenEnough += float(env_info["open_enough"])
        if terminated or truncated:
            # normalize by trajectory length to get fraction of trajectory
            # where True
            self.EECloseToHandle /= self.Length
            self.OpenEnough /= self.Length


@dataclass
class MetaworldTrajInfo(TrajInfo):
    Success: bool = False
    SuccessLength: int = 0

    def step(
        self,
        observation: ObsType,
        action: ActionType,
        reward: RewardType,
        terminated: DoneType,
        truncated: DoneType,
        env_info: EnvInfoType,
    ) -> None:
        super().step(observation, action, reward, terminated, truncated, env_info)
        self.Success = env_info["success"] or self.Success
        if not self.Success:
            self.SuccessLength += 1
