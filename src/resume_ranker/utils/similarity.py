from __future__ import annotations
import numpy as np
from typing import Tuple


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: [N, d], b: [M, d] -> [N, M]
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("输入维度不匹配。")
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.matmul(a_norm, b_norm.T)


def topk_indices(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每一行取 top-k，返回 (indices, values)
    """
    if k <= 0:
        raise ValueError("k 必须为正整数。")
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part_vals, axis=1)
    top_idx = np.take_along_axis(idx, order, axis=1)
    top_vals = np.take_along_axis(part_vals, order, axis=1)
    return top_idx, top_vals
