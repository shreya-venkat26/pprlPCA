from __future__ import annotations

from typing import Sequence

import gymnasium.spaces as spaces
import torch
import torch.nn as nn
from parllel import ArrayTree
from torch import Tensor
from torch.nn import Linear as Lin
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    global_mean_pool,
    knn,
    knn_graph,
)
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import scatter

from pprl.envs import PointCloudSpace
from pprl.utils.array_dict import dict_to_batched_data

if not WITH_TORCH_CLUSTER:
    raise ImportError(
        "PointNet++ requires fps. Install torch geometric with cuda support."
    )


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP(
            [out_channels, 64, out_channels], norm=None, plain_last=False
        )

        self.transformer = PointTransformerConv(
            in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality.
    """

    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(
            pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch
        )

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(
            x[id_k_neighbor[1]],
            id_k_neighbor[0],
            dim=0,
            dim_size=id_clusters.size(0),
            reduce="max",
        )

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class PointTransformer(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Space,
        embed_dim: int,
        dim_model: Sequence[int],
        k: int = 16,
        state_embed_dim: int | None = None,
    ):
        super().__init__()
        self.k = k

        if obs_is_dict := isinstance(obs_space, spaces.Dict):
            point_space = obs_space["points"]
        else:
            point_space = obs_space
        assert isinstance(point_space, PointCloudSpace)
        self.obs_is_dict = obs_is_dict

        point_dim = point_space.shape[0]
        in_channels = point_dim - 3  # in_channels are only color
        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0], out_channels=dim_model[0]
        )
        # backbone layers
        self.transformers_down = nn.ModuleList()
        self.transition_down = nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(
                    in_channels=dim_model[i], out_channels=dim_model[i + 1], k=self.k
                )
            )

            self.transformers_down.append(
                TransformerBlock(
                    in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]
                )
            )

        # class score computation
        self.mlp_output = MLP([dim_model[-1], 64, embed_dim], norm=None)

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

    @property
    def embed_dim(self) -> int:
        return self.mlp_output.out_channels + self.state_dim

    def forward(self, observation: ArrayTree[Tensor]) -> Tensor:
        point_cloud: ArrayTree[Tensor] = (
            observation["points"] if self.obs_is_dict else observation
        )
        pos, batch, x = dict_to_batched_data(point_cloud)  # type: ignore

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)
        encoder_out = self.mlp_output(x)

        if self.obs_is_dict:
            state = [elem for name, elem in observation.items() if name != "points"]
            if self.state_encoder is not None:
                state = torch.concatenate(state, dim=-1)
                state = self.state_encoder(state)
                state = [state]

            encoder_out = torch.concatenate([encoder_out] + state, dim=-1)

        return encoder_out
