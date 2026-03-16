"""Patch dataset for round-specific training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class PreparedCase:
    case_id: str
    image: np.ndarray
    target: np.ndarray
    voxel_weight: np.ndarray
    case_weight: float


class RoundPatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        cases: list[PreparedCase],
        patch_size: tuple[int, int, int],
        patches_per_case: int,
        positive_patch_probability: float,
        seed: int,
        augmentation_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.cases = cases
        self.patch_size = patch_size
        self.patches_per_case = patches_per_case
        self.positive_patch_probability = positive_patch_probability
        self.rng = np.random.default_rng(seed)
        augmentation_cfg = augmentation_cfg or {}
        self.flip_axes = tuple(int(axis) for axis in augmentation_cfg.get("flip_axes", []))
        self.intensity_scale_range = tuple(float(v) for v in augmentation_cfg.get("intensity_scale_range", (1.0, 1.0)))
        self.intensity_shift_range = tuple(float(v) for v in augmentation_cfg.get("intensity_shift_range", (0.0, 0.0)))
        self.gaussian_noise_std = float(augmentation_cfg.get("gaussian_noise_std", 0.0))

    def __len__(self) -> int:
        return max(1, len(self.cases) * self.patches_per_case)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case = self.cases[index % len(self.cases)]
        image_patch, target_patch, weight_patch = self._sample_patch(case)
        image_patch, target_patch, weight_patch = self._apply_augmentations(image_patch, target_patch, weight_patch)
        return {
            "image": torch.from_numpy(image_patch[None]),
            "target": torch.from_numpy(target_patch[None]),
            "voxel_weight": torch.from_numpy(weight_patch[None]),
            "case_weight": torch.tensor(case.case_weight, dtype=torch.float32),
        }

    def _sample_patch(self, case: PreparedCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        spatial_shape = case.image.shape
        patch = np.asarray(self.patch_size)
        starts = np.zeros(3, dtype=np.int32)
        choose_positive = self.rng.random() < self.positive_patch_probability and case.target.sum() > 0
        if choose_positive:
            positive_indices = np.argwhere(case.target > 0.5)
            center = positive_indices[self.rng.integers(len(positive_indices))]
            starts = center - patch // 2
        else:
            max_starts = np.maximum(np.asarray(spatial_shape) - patch, 0)
            starts = np.asarray([self.rng.integers(m + 1) if m > 0 else 0 for m in max_starts], dtype=np.int32)
        starts = np.clip(starts, 0, np.maximum(np.asarray(spatial_shape) - patch, 0))
        ends = starts + patch
        slices = tuple(slice(int(starts[idx]), int(ends[idx])) for idx in range(3))
        return (
            case.image[slices].astype(np.float32, copy=False),
            case.target[slices].astype(np.float32, copy=False),
            case.voxel_weight[slices].astype(np.float32, copy=False),
        )

    def _apply_augmentations(
        self,
        image_patch: np.ndarray,
        target_patch: np.ndarray,
        weight_patch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_patch = image_patch.astype(np.float32, copy=True)
        target_patch = target_patch.astype(np.float32, copy=True)
        weight_patch = weight_patch.astype(np.float32, copy=True)

        for axis in self.flip_axes:
            if self.rng.random() < 0.5:
                image_patch = np.flip(image_patch, axis=axis).copy()
                target_patch = np.flip(target_patch, axis=axis).copy()
                weight_patch = np.flip(weight_patch, axis=axis).copy()

        if len(self.intensity_scale_range) == 2:
            low, high = self.intensity_scale_range
            if high > low:
                image_patch *= np.float32(self.rng.uniform(low, high))

        if len(self.intensity_shift_range) == 2:
            low, high = self.intensity_shift_range
            if high > low:
                image_patch += np.float32(self.rng.uniform(low, high))

        if self.gaussian_noise_std > 0.0:
            image_patch += self.rng.normal(0.0, self.gaussian_noise_std, size=image_patch.shape).astype(np.float32)

        return image_patch, target_patch, weight_patch
