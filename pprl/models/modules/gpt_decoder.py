from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .transformer import TransformerDecoder


class GPTDecoder(nn.Module):
    def __init__(
        self,
        transformer_decoder: Callable,
        pos_embedder: Callable,
        embed_dim: int,
        padding_value: float = -1.0,
        absolute_pos: bool = False,
    ):
        super().__init__()
        self.transformer_decoder: TransformerDecoder = transformer_decoder(
            embed_dim=embed_dim
        )
        self.pos_embedder: nn.Module = pos_embedder(token_dim=embed_dim)
        self.padding_value = padding_value
        self.norm = nn.LayerNorm(self.embed_dim)
        self.absolute_pos = absolute_pos
        self.apply(self._init_weights)

    @property
    def embed_dim(self) -> int:
        return self.transformer_decoder.embed_dim

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, center_points, padding_mask=None, attn_mask=None):
        relative_pos = center_points[:, 1:, :] - center_points[:, :-1, :]
        if self.absolute_pos:
            relative_direction = torch.cat(
                [center_points[:, 0, :].unsqueeze(1), relative_pos], dim=1
            )

        else:
            relative_pos_norm = torch.linalg.vector_norm(
                relative_pos, dim=-1, keepdim=True
            )
            relative_direction = torch.nan_to_num(relative_pos / (relative_pos_norm))
            # prepend absolute position of first center point
            relative_direction = torch.cat(
                [center_points[:, 0, :].unsqueeze(1), relative_direction], dim=1
            )
        relative_pos = self.pos_embedder(relative_direction)
        x = self.transformer_decoder(
            x, relative_pos, padding_mask=padding_mask, attn_mask=attn_mask
        )
        x = self.norm(x)
        return x
