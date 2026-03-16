"""Protocol metrics and uncertainty helpers."""

from __future__ import annotations

from math import log

import numpy as np
from scipy import ndimage


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * np.logical_and(a, b).sum()) / denom)


def soft_dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1.0e-6) -> float:
    intersection = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    return (2.0 * intersection + eps) / (denom + eps)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1.0e-8, 1.0 - 1.0e-8)
    return (-clipped * np.log(clipped) - (1.0 - clipped) * np.log(1.0 - clipped)) / log(2)


def fused_uncertainty_map(soft_target: np.ndarray, alpha: float) -> np.ndarray:
    normalizer = float(binary_entropy(np.asarray([alpha], dtype=np.float32))[0])
    return (binary_entropy(soft_target) / normalizer).astype(np.float32)


def top_n_mean(values: np.ndarray, n: int) -> float:
    flat = values.reshape(-1)
    if flat.size == 0:
        return 0.0
    n = max(1, min(n, flat.size))
    idx = np.argpartition(flat, flat.size - n)[-n:]
    return float(flat[idx].mean())


def connected_components_score(binary_mask: np.ndarray, max_components: int = 20) -> tuple[int, float]:
    _, num = ndimage.label(binary_mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
    clipped = min(int(num), max_components)
    return int(num), clipped / max_components


def positive_slice_fraction(binary_mask: np.ndarray) -> float:
    if binary_mask.ndim != 3:
        raise ValueError("Expected a 3D mask")
    positives = np.any(binary_mask.astype(bool), axis=(0, 1))
    return float(positives.mean())


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def edit_ratio(current_mask: np.ndarray, previous_mask: np.ndarray) -> float:
    numerator = np.abs(current_mask.astype(np.int16) - previous_mask.astype(np.int16)).sum()
    denominator = np.logical_or(current_mask.astype(bool), previous_mask.astype(bool)).sum()
    return float(numerator / (denominator + 1.0e-6))

