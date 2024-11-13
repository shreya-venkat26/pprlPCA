import numpy as np
import open3d as o3d
from open3d.geometry import PointCloud


def np_to_o3d(array: np.ndarray) -> PointCloud:
    assert (shape := array.shape[-1]) in (3, 6)
    pos = array[:, :3]
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pos))
    if shape == 6:
        color = array[:, 3:]
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def o3d_to_np(pcd: PointCloud) -> np.ndarray:
    if pcd.has_colors():
        return np.hstack(
            (
                np.asarray(pcd.points, dtype=np.float32),
                np.asarray(pcd.colors, dtype=np.float32),
            )
        )
    else:
        return np.asarray(pcd.points, dtype=np.float32)
