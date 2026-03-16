"""Probability-to-mask postprocessing helpers."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def threshold_probability(probability: np.ndarray, threshold: float) -> np.ndarray:
    return (probability >= float(threshold)).astype(np.uint8)


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(binary_mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
    if num == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    component_sizes = ndimage.sum(binary_mask.astype(np.uint8), labeled, index=np.arange(1, num + 1))
    largest_index = int(np.argmax(component_sizes)) + 1
    return (labeled == largest_index).astype(np.uint8)


def remove_small_components(binary_mask: np.ndarray, min_component_voxels: int) -> np.ndarray:
    if min_component_voxels <= 1:
        return binary_mask.astype(np.uint8, copy=False)
    labeled, num = ndimage.label(binary_mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
    if num == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    component_sizes = ndimage.sum(binary_mask.astype(np.uint8), labeled, index=np.arange(1, num + 1))
    keep = np.zeros(num + 1, dtype=bool)
    for idx, size in enumerate(component_sizes, start=1):
        if int(size) >= min_component_voxels:
            keep[idx] = True
    return keep[labeled].astype(np.uint8)


def postprocess_probability_map(
    probability: np.ndarray,
    threshold: float,
    min_component_voxels: int = 0,
    largest_only: bool = False,
) -> np.ndarray:
    binary_mask = threshold_probability(probability, threshold)
    binary_mask = remove_small_components(binary_mask, min_component_voxels)
    if largest_only:
        binary_mask = keep_largest_component(binary_mask)
    return binary_mask.astype(np.uint8, copy=False)
