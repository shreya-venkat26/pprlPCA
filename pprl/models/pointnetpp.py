from __future__ import annotations

import gymnasium.spaces as spaces
import torch
import torch.nn as nn
from parllel import ArrayTree
from torch import Tensor
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.typing import WITH_TORCH_CLUSTER

from pprl.envs import PointCloudSpace
from pprl.utils.array_dict import dict_to_batched_data

if not WITH_TORCH_CLUSTER:
    raise ImportError(
        "PointNet++ requires fps. Install torch geometric with cuda support."
    )


class SAModule(nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNetPP(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Space,
        embed_dim: int,
        dropout: float = 0.0,
        state_embed_dim: int | None = None,
    ):
        super().__init__()

        if obs_is_dict := isinstance(obs_space, spaces.Dict):
            point_space = obs_space["points"]
        else:
            point_space = obs_space
        assert isinstance(point_space, PointCloudSpace)
        self.obs_is_dict = obs_is_dict

        point_dim = point_space.shape[0]

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([point_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, embed_dim], dropout=dropout, norm=None)

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
        sa0_out = (color, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        encoder_out = self.mlp(x)

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
        return self.mlp.out_channels + self.state_dim
