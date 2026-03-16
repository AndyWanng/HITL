"""Sliding-window inference and TTA helpers."""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
import torch


def iter_sliding_windows(shape: tuple[int, int, int], patch_size: tuple[int, int, int], overlap: float) -> Iterable[tuple[slice, slice, slice]]:
    strides = [max(1, int(size * (1.0 - overlap))) for size in patch_size]
    axes: list[list[int]] = []
    for dim, patch, stride in zip(shape, patch_size, strides, strict=True):
        if dim <= patch:
            axes.append([0])
            continue
        starts = list(range(0, dim - patch + 1, stride))
        if starts[-1] != dim - patch:
            starts.append(dim - patch)
        axes.append(starts)
    for start_x, start_y, start_z in product(*axes):
        yield (
            slice(start_x, start_x + patch_size[0]),
            slice(start_y, start_y + patch_size[1]),
            slice(start_z, start_z + patch_size[2]),
        )


def sliding_window_predict(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device,
    patch_size: tuple[int, int, int],
    overlap: float,
) -> np.ndarray:
    model.eval()
    accum = np.zeros(volume.shape, dtype=np.float32)
    counts = np.zeros(volume.shape, dtype=np.float32)
    with torch.no_grad():
        for slices in iter_sliding_windows(volume.shape, patch_size, overlap):
            patch = volume[slices][None, None]
            patch_tensor = torch.from_numpy(patch).to(device=device, dtype=torch.float32)
            logits = model(patch_tensor)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            accum[slices] += probs
            counts[slices] += 1.0
    counts[counts == 0] = 1.0
    return accum / counts


def apply_tta(volume: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return volume
    if mode == "flip_x":
        return np.flip(volume, axis=0).copy()
    if mode == "flip_y":
        return np.flip(volume, axis=1).copy()
    if mode == "flip_xy":
        return np.flip(np.flip(volume, axis=0), axis=1).copy()
    raise ValueError(f"Unsupported TTA mode: {mode}")


def invert_tta(volume: np.ndarray, mode: str) -> np.ndarray:
    return apply_tta(volume, mode)
