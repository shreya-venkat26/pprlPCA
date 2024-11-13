from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        attention: MultiheadAttention | Callable,
        mlp: nn.Module | Callable,
        embed_dim: int | None = None,
        NormLayer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        if isinstance(attention, nn.Module):
            self.attention = attention
        else:
            if not embed_dim:
                raise ValueError(f"{embed_dim=}")
            self.attention = attention(embed_dim=embed_dim)

        if isinstance(mlp, nn.Module):
            self.mlp = mlp
        else:
            if not embed_dim:
                raise ValueError(f"{embed_dim=}")
            self.mlp = mlp(embed_dim=embed_dim)

        self.norm_1 = NormLayer(self.dim)
        self.norm_2 = NormLayer(self.dim)
        self._check_mlp()

    @property
    def dim(self) -> int:
        return self.attention.embed_dim

    def _check_mlp(self):
        with torch.no_grad():
            input = torch.randn((self.dim,))
            try:
                out = self.mlp(input)
                out = self.norm_1(out)
            except RuntimeError as e:
                raise ValueError(
                    f"The first and the layer of the MLP must have the same size as the embedding_dim: {self.dim}"
                ) from e

    def forward(self, x, padding_mask=None, attn_mask=None):
        x = self.norm_1(x)
        x = (
            x
            + self.attention(
                x,
                x,
                x,
                need_weights=False,
                key_padding_mask=padding_mask,
                attn_mask=attn_mask,
            )[0]
        )
        x = self.norm_2(x)
        x = x + self.mlp(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, block_factory: Callable, embed_dim: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [block_factory(embed_dim=embed_dim) for _ in range(depth)]
        )

    @property
    def embed_dim(self) -> int:
        return self.blocks[0].dim

    def forward(self, x, pos, padding_mask=None, attn_mask=None):
        for block in self.blocks:
            x = block(x + pos, padding_mask=padding_mask, attn_mask=attn_mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        block_factory: Callable,
        embed_dim: int,
        depth: int,
        NormLayer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [block_factory(embed_dim=embed_dim) for _ in range(depth)]
        )
        self.norm = NormLayer(embed_dim)

        self.apply(self._init_weights)

    @property
    def embed_dim(self) -> int:
        return self.blocks[0].dim

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num=None, padding_mask=None, attn_mask=None):
        for block in self.blocks:
            x = block(x + pos, padding_mask=padding_mask, attn_mask=attn_mask)

        if return_token_num is not None:
            x = x[:, -return_token_num:]
        x = self.norm(x)
        return x


class SequencePooling(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention_pool = nn.Linear(embed_dim, 1)

    @property
    def embed_dim(self) -> int:
        return self.attention_pool.in_features

    def forward(self, x):
        # TODO: isn't this norm redundant?
        x = self.norm(x)

        weights = F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2)
        x = torch.matmul(weights, x).squeeze(-2)
        return x
