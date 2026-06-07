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
    alignment_core_mask: np.ndarray | None = None
    alignment_core_target: np.ndarray | None = None
    alignment_core_weight: np.ndarray | None = None
    alignment_sampling_mask: np.ndarray | None = None
    alignment_teacher: np.ndarray | None = None
    alignment_trust_weight: np.ndarray | None = None


class RoundPatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        cases: list[PreparedCase],
        patch_size: tuple[int, int, int],
        patches_per_case: int,
        positive_patch_probability: float,
        seed: int,
        augmentation_cfg: dict[str, Any] | None = None,
        alignment_core_patch_probability: float = 0.0,
        reviewed_positive_patch_probability: float | None = None,
    ) -> None:
        self.cases = cases
        self.patch_size = patch_size
        self.patches_per_case = patches_per_case
        self.positive_patch_probability = positive_patch_probability
        self.alignment_core_patch_probability = alignment_core_patch_probability
        self.reviewed_positive_patch_probability = reviewed_positive_patch_probability
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
        (
            image_patch,
            target_patch,
            weight_patch,
            alignment_core_mask_patch,
            alignment_core_target_patch,
            alignment_core_weight_patch,
            alignment_teacher_patch,
            alignment_trust_weight_patch,
        ) = self._sample_patch(case)
        (
            image_patch,
            target_patch,
            weight_patch,
            alignment_core_mask_patch,
            alignment_core_target_patch,
            alignment_core_weight_patch,
            alignment_teacher_patch,
            alignment_trust_weight_patch,
        ) = self._apply_augmentations(
            image_patch,
            target_patch,
            weight_patch,
            alignment_core_mask_patch,
            alignment_core_target_patch,
            alignment_core_weight_patch,
            alignment_teacher_patch,
            alignment_trust_weight_patch,
        )
        return {
            "image": torch.from_numpy(image_patch[None]),
            "target": torch.from_numpy(target_patch[None]),
            "voxel_weight": torch.from_numpy(weight_patch[None]),
            "case_weight": torch.tensor(case.case_weight, dtype=torch.float32),
            "alignment_core_mask": torch.from_numpy(alignment_core_mask_patch[None]),
            "alignment_core_target": torch.from_numpy(alignment_core_target_patch[None]),
            "alignment_core_weight": torch.from_numpy(alignment_core_weight_patch[None]),
            "alignment_teacher": torch.from_numpy(alignment_teacher_patch[None]),
            "alignment_trust_weight": torch.from_numpy(alignment_trust_weight_patch[None]),
        }

    def _sample_patch(self, case: PreparedCase) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        spatial_shape = case.image.shape
        patch = np.asarray(self.patch_size)
        starts = np.zeros(3, dtype=np.int32)
        alignment_sampling_mask = self._case_array(case, "alignment_sampling_mask", fallback_attr="alignment_core_mask")
        has_alignment_core = alignment_sampling_mask.sum() > 0
        draw = self.rng.random()
        if has_alignment_core and draw < self.alignment_core_patch_probability:
            core_indices = np.argwhere(alignment_sampling_mask > 0)
            center = core_indices[self.rng.integers(len(core_indices))]
            starts = center - patch // 2
        elif self._choose_positive(case, has_alignment_core, draw):
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
            self._case_array(case, "alignment_core_mask")[slices].astype(np.float32, copy=False),
            self._case_array(case, "alignment_core_target")[slices].astype(np.float32, copy=False),
            self._case_array(case, "alignment_core_weight")[slices].astype(np.float32, copy=False),
            self._case_array(case, "alignment_teacher")[slices].astype(np.float32, copy=False),
            self._case_array(case, "alignment_trust_weight")[slices].astype(np.float32, copy=False),
        )

    def _case_array(self, case: PreparedCase, attr: str, fallback_attr: str | None = None) -> np.ndarray:
        value = getattr(case, attr)
        if value is None and fallback_attr is not None:
            value = getattr(case, fallback_attr)
        if value is None:
            return np.zeros_like(case.target, dtype=np.float32)
        return value.astype(np.float32, copy=False)

    def _choose_positive(self, case: PreparedCase, has_alignment_core: bool, draw: float) -> bool:
        if not np.any(case.target > 0.5):
            return False
        if has_alignment_core and self.reviewed_positive_patch_probability is not None:
            upper = self.alignment_core_patch_probability + self.reviewed_positive_patch_probability
            return draw < upper
        return draw < self.positive_patch_probability

    def _apply_augmentations(
        self,
        image_patch: np.ndarray,
        target_patch: np.ndarray,
        weight_patch: np.ndarray,
        alignment_core_mask_patch: np.ndarray,
        alignment_core_target_patch: np.ndarray,
        alignment_core_weight_patch: np.ndarray,
        alignment_teacher_patch: np.ndarray,
        alignment_trust_weight_patch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image_patch = image_patch.astype(np.float32, copy=True)
        target_patch = target_patch.astype(np.float32, copy=True)
        weight_patch = weight_patch.astype(np.float32, copy=True)
        alignment_core_mask_patch = alignment_core_mask_patch.astype(np.float32, copy=True)
        alignment_core_target_patch = alignment_core_target_patch.astype(np.float32, copy=True)
        alignment_core_weight_patch = alignment_core_weight_patch.astype(np.float32, copy=True)
        alignment_teacher_patch = alignment_teacher_patch.astype(np.float32, copy=True)
        alignment_trust_weight_patch = alignment_trust_weight_patch.astype(np.float32, copy=True)

        for axis in self.flip_axes:
            if self.rng.random() < 0.5:
                image_patch = np.flip(image_patch, axis=axis).copy()
                target_patch = np.flip(target_patch, axis=axis).copy()
                weight_patch = np.flip(weight_patch, axis=axis).copy()
                alignment_core_mask_patch = np.flip(alignment_core_mask_patch, axis=axis).copy()
                alignment_core_target_patch = np.flip(alignment_core_target_patch, axis=axis).copy()
                alignment_core_weight_patch = np.flip(alignment_core_weight_patch, axis=axis).copy()
                alignment_teacher_patch = np.flip(alignment_teacher_patch, axis=axis).copy()
                alignment_trust_weight_patch = np.flip(alignment_trust_weight_patch, axis=axis).copy()

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

        return (
            image_patch,
            target_patch,
            weight_patch,
            alignment_core_mask_patch,
            alignment_core_target_patch,
            alignment_core_weight_patch,
            alignment_teacher_patch,
            alignment_trust_weight_patch,
        )
