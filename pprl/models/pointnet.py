from __future__ import annotations

import gymnasium.spaces as spaces
import torch
import torch.nn as nn
from parllel import ArrayTree
from torch import Tensor
from torch_geometric.nn import MLP, global_max_pool

from pprl.envs import PointCloudSpace
from pprl.utils.array_dict import dict_to_batched_data


class PointNet(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Space,
        embed_dim: int,
        state_embed_dim: int | None = None,
    ) -> None:
        super().__init__()

        if obs_is_dict := isinstance(obs_space, spaces.Dict):
            point_space = obs_space["points"]
        else:
            point_space = obs_space
        assert isinstance(point_space, PointCloudSpace)
        self.obs_is_dict = obs_is_dict

        point_dim = point_space.shape[0]
        self.mlp1 = MLP([point_dim, 128, 256, 512], norm="layer_norm", plain_last=False)
        self.mlp2 = MLP([512, embed_dim])

        self.state_encoder = None
        if self.obs_is_dict:
            state_dim = sum(
                space.shape[0] for name, space in obs_space.items() if name != "points"
            )
            # maybe create linear projection layer for state vector
            if state_embed_dim is not None:
                self.state_encoder = nn.Linear(state_dim, state_embed_dim)
                self.state_dim = state_embed_dim
            else:
                self.state_dim = state_dim
        else:
            self.state_dim = 0

    def forward(self, observation: ArrayTree[Tensor]) -> Tensor:
        point_cloud: ArrayTree[Tensor] = (
            observation["points"] if self.obs_is_dict else observation
        )
        pos, batch, color = dict_to_batched_data(point_cloud)  # type: ignore
        input = pos if color is None else torch.hstack((pos, color))
        x = self.mlp1(input)
        x = global_max_pool(x, batch)
        encoder_out = self.mlp2(x)

        if self.obs_is_dict:
            state = [elem for name, elem in observation.items() if name != "points"]
            if self.state_encoder is not None:
                state = torch.concatenate(state, dim=-1)
                state = self.state_encoder(state)
                state = [state]

            encoder_out = torch.concatenate([encoder_out] + state, dim=-1)

        return encoder_out

    @property
    def embed_dim(self) -> int:
        return self.mlp2.out_channels + self.state_dim
