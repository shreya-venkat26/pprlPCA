from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import TransformerDecoder


class MaskedDecoder(nn.Module):
    def __init__(
        self,
        transformer_decoder: TransformerDecoder,
        pos_embedder: nn.Module,
        embed_dim: int,
        padding_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.transformer_decoder: TransformerDecoder = transformer_decoder(
            embed_dim=embed_dim
        )
        self.pos_embedder: nn.Module = pos_embedder(token_dim=embed_dim)
        self.padding_value = padding_value

        # register mask token as a buffer so it gets put onto the same device
        mask_token = torch.empty(self.embed_dim)
        nn.init.trunc_normal_(mask_token, std=0.02)
        self.register_buffer("mask_token", mask_token)

        # TODO: weight initialization?

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

    def forward(self, x_vis, mask, center_points):
        B, _, C = x_vis.shape
        center_points_visible = center_points[~mask].reshape(B, -1, 3)
        center_points_masked = center_points[mask].reshape(B, -1, 3)
        _, num_masked_tokens, _ = center_points_masked.shape

        center_points_full = torch.cat(
            [center_points_visible, center_points_masked], dim=1
        )
        # since we reordered the center points, we have to recalculate the padding mask
        padding_mask = torch.all(center_points_full == self.padding_value, dim=-1)

        pos_full = self.pos_embedder(center_points_full).reshape(B, -1, C)

        mask_token = self.mask_token.expand(B, num_masked_tokens, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)

        x_recovered = self.transformer_decoder(
            x_full, pos_full, num_masked_tokens, padding_mask
        )

        return x_recovered
