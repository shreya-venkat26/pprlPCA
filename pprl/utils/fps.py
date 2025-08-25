import numpy as np

def farthest_point_sampling(points: np.ndarray, num_samples: int, init_idx: int | None = None) -> np.ndarray:
    """
    Fast Farthest Point Sampling (FPS) from a point cloud.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) containing point coordinates.
    num_samples : int
        Number of points to sample.
    init_idx : int or None, optional
        Optional starting index for the sampling. If ``None`` a random
        point is used as the initial point.

    Returns
    -------
    np.ndarray
        Indices of the sampled points with shape ``(num_samples,)``.
    """
    N = points.shape[0]
    sampled_idxs = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(N, np.inf)
    if init_idx is None:
        init_idx = np.random.randint(N)
    sampled_idxs[0] = init_idx
    current_point = points[init_idx]
    for i in range(1, num_samples):
        dist = np.sum((points - current_point) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        idx = np.argmax(distances)
        sampled_idxs[i] = idx
        current_point = points[idx]
    return sampled_idxs