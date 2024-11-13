from __future__ import annotations

from typing import Sequence

from torch_geometric.nn import MLP


def TokenizerMLP(
    hidden_layers: Sequence[int],
    input_size: int | None = None,
    output_size: int | None = None,
    **kwargs,
) -> MLP:
    mlp_layers = (
        ([input_size] if input_size else [])
        + list(hidden_layers)
        + ([output_size] if output_size else [])
    )
    return MLP(mlp_layers, **kwargs)


def TransformerBlockMLP(embed_dim: int, mlp_ratio: float, **kwargs) -> MLP:
    mlp_layers = [embed_dim, int(mlp_ratio * embed_dim), embed_dim]
    return MLP(mlp_layers, **kwargs)


def PositionalEmbedderMLP(
    n_dim: int, hidden_layers: Sequence[int], token_dim: int, **kwargs
) -> MLP:
    mlp_layers = [n_dim] + list(hidden_layers) + [token_dim]
    return MLP(mlp_layers, **kwargs)
