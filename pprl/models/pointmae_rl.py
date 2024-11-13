from __future__ import annotations

from typing import Callable

from parllel import ArrayTree
from torch import Tensor

from pprl.models.modules.masked_decoder import MaskedDecoder
from pprl.models.modules.masked_encoder import MaskedEncoder
from pprl.models.modules.prediction_head import PredictionHead
from pprl.models.ppt import PointPatchTransformer
from pprl.utils.array_dict import dict_to_batched_data


class PointMAE(PointPatchTransformer):
    def __init__(
        self,
        masked_encoder: Callable,
        masked_decoder: Callable,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # remove these from the module so that the parameters are not registered twice
        transformer_encoder = self.transformer_encoder
        del self.transformer_encoder
        pos_embedder = self.pos_embedder
        del self.pos_embedder

        self.masked_encoder: MaskedEncoder = masked_encoder(
            transformer_encoder=transformer_encoder, pos_embedder=pos_embedder
        )
        self.masked_decoder: MaskedDecoder = masked_decoder(
            embed_dim=transformer_encoder.embed_dim
        )
        self.prediction_head = PredictionHead(
            dim=self.tokenizer.embed_dim,
            group_size=self.tokenizer.group_size,
            point_dim=self.tokenizer.point_dim,
        )

        # Dynamically create properties so that the parent class can access these
        # modules as if they were still there in its forward method.
        # We can't simply assign them as properties, since they would get registered.
        # Properties can be created dynamically, but they must be assigned to
        # the class, not the instance: https://stackoverflow.com/a/1355444
        PointMAE.transformer_encoder = property(
            lambda self: self.masked_encoder.transformer_encoder
        )
        PointMAE.pos_embedder = property(lambda self: self.masked_encoder.pos_embedder)

    def reconstruct(self, observation: dict):
        point_cloud: ArrayTree[Tensor] = (
            observation["points"] if self.obs_is_dict else observation
        )
        pos, batch, color = dict_to_batched_data(point_cloud)

        x, neighborhoods, center_points = self.tokenizer(pos, batch, color)

        x_vis, ae_mask, padding_mask = self.masked_encoder(x, center_points)
        x_recovered = self.masked_decoder(x_vis, ae_mask, center_points)
        prediction = self.prediction_head(x_recovered)
        B, M, *_ = x_recovered.shape
        *_, C = neighborhoods.shape

        padding_mask = padding_mask.reshape(B, -1)
        padding_mask = padding_mask[ae_mask].reshape(B, -1)

        ground_truth = neighborhoods[ae_mask].reshape(B, M, -1, C)
        prediction = prediction.reshape(B, M, -1, C)

        ground_truth[padding_mask] = 0.0
        prediction[padding_mask] = 0.0

        return prediction, ground_truth
