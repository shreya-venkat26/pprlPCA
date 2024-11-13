from __future__ import annotations

import torch.nn.functional as F
from parllel import ArrayDict
from parllel.torch.algos.sac import SAC
from pytorch3d.ops.knn import knn_gather
from torch import Tensor

from pprl.utils.chamfer import chamfer_distance


class AuxPcSAC(SAC):
    def __init__(
        self,
        chamfer_loss_coeff: float,
        color_loss_coeff: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # TODO: add scheduler, like a learning rate scheduler, which stops loss
        # computation if the value is 0
        self.chamfer_loss_coeff = chamfer_loss_coeff
        self.color_loss_coeff = color_loss_coeff

    def critic_loss(self, samples: ArrayDict[Tensor]) -> Tensor:
        q_loss = super().critic_loss(samples)

        observation = samples["observation"]
        prediction, ground_truth = self.agent.model["encoder"].reconstruct(observation)

        B, M, *_, C = prediction.shape
        prediction = prediction.reshape(B * M, -1, C)
        ground_truth = ground_truth.reshape(B * M, -1, C)

        chamfer_loss, _, x_idx = chamfer_distance(  # type: ignore
            prediction[..., :3],
            ground_truth[..., :3],
            return_x_nn=True,
        )
        chamfer_loss *= self.chamfer_loss_coeff
        self.algo_log_info["chamfer_loss"].append(chamfer_loss.item())

        # if color exists and color loss is requested
        if self.color_loss_coeff is not None and C > 3:
            assert x_idx is not None
            prediction_nearest_neighbor = knn_gather(ground_truth, x_idx).reshape(
                B, M, -1, C
            )
            color_loss = (
                F.mse_loss(
                    prediction[..., 3:].reshape(B, M, -1, C - 3),
                    prediction_nearest_neighbor[..., 3:].reshape(B, M, -1, C - 3),
                )
                * self.color_loss_coeff
            )
            self.algo_log_info["color_loss"].append(color_loss.item())
            chamfer_loss += color_loss

        return q_loss + chamfer_loss
