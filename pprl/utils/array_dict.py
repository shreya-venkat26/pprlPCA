from __future__ import annotations

import gymnasium.spaces as spaces
import numpy as np
import torch
from parllel import Array, ArrayDict, ArrayOrMapping
from torch import Tensor

from pprl.envs import PointCloudSpace


def dict_to_batched_data(
    array_dict: ArrayDict[Tensor],
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Convert an ArrayDict with `pos` and `ptr` Tensors to pyg convention
    with `pos`, `batch`, and `feature` Tensors.
    """
    points, ptr = array_dict["pos"], array_dict["ptr"]
    num_nodes = ptr[1:] - ptr[:-1]
    pos = points[..., :3]
    features = points[..., 3:] if points.shape[-1] > 3 else None
    batch = torch.repeat_interleave(
        torch.arange(len(num_nodes), device=num_nodes.device),
        repeats=num_nodes,
    )
    return pos, batch, features

def dict_to_batched_data_pca(
    points, ptr
):
    """Convert an ArrayDict with `pos` and `ptr` Tensors to pyg convention
    with `pos`, `batch`, and `feature` Tensors.
    """

    num_nodes = ptr[1:] - ptr[:-1]
    pos = points[..., :3]
    # features = points[..., 3:] if points.shape[-1] > 3 else None
    if points.shape[-1] > 3:
        print("FEATURES")
        exit()

    batch = torch.repeat_interleave(
        torch.arange(len(num_nodes), device=num_nodes.device),
        repeats=num_nodes,
    )
    return pos, batch


def build_obs_array(
    example_obs: ArrayOrMapping[np.ndarray],
    obs_space: spaces.Space,
    **kwargs,
) -> ArrayOrMapping[Array]:
    if isinstance(example_obs, dict):
        d = {}
        for name, elem in example_obs.items():
            elem_space = obs_space[name]
            d[name] = build_obs_array(elem, elem_space, **kwargs)
        return ArrayDict(d)

    # single element
    if isinstance(obs_space, PointCloudSpace):
        kwargs = kwargs.copy()  # don't modify kwargs in-place
        kwargs["kind"] = "jagged"
        kwargs["max_mean_num_elem"] = obs_space.max_expected_num_points

    return Array.from_numpy(example_obs, **kwargs)
