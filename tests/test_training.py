from __future__ import annotations

import unittest

import numpy as np
import torch

from hemorrhage.training.dataset import PreparedCase, RoundPatchDataset
from hemorrhage.training.losses import protocol_loss


class TrainingLossTests(unittest.TestCase):
    def test_protocol_loss_backprop(self) -> None:
        logits = torch.zeros((2, 1, 8, 8, 8), requires_grad=True)
        target = torch.ones((2, 1, 8, 8, 8)) * 0.5
        voxel_weight = torch.ones_like(target)
        case_weight = torch.tensor([1.0, 2.0])
        loss = protocol_loss(logits, target, voxel_weight, case_weight)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_protocol_loss_with_alignment_backprop(self) -> None:
        logits = torch.zeros((2, 1, 8, 8, 8), requires_grad=True)
        target = torch.zeros((2, 1, 8, 8, 8))
        target[:, :, 4, 4, 4] = 1.0
        voxel_weight = torch.ones_like(target)
        case_weight = torch.tensor([1.0, 2.0])
        alignment_core_mask = torch.zeros_like(target)
        alignment_core_target = torch.zeros_like(target)
        alignment_core_weight = torch.zeros_like(target)
        alignment_teacher = torch.zeros_like(target) + 0.01
        alignment_trust_weight = torch.ones_like(target)
        alignment_trust_weight[:, :, 4, 4, 4] = 0.0
        alignment_core_mask[:, :, 4, 4, 4] = 1.0
        alignment_core_target[:, :, 4, 4, 4] = 1.0
        alignment_core_weight[:, :, 4, 4, 4] = 0.9
        loss = protocol_loss(
            logits,
            target,
            voxel_weight,
            case_weight,
            alignment_core_mask=alignment_core_mask,
            alignment_core_target=alignment_core_target,
            alignment_core_weight=alignment_core_weight,
            alignment_teacher=alignment_teacher,
            alignment_trust_weight=alignment_trust_weight,
            alignment_core_loss_weight=0.08,
            alignment_distill_loss_weight=0.15,
            alignment_volume_guard_weight=0.05,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_protocol_loss_ignores_empty_alignment_masks(self) -> None:
        logits = torch.randn((1, 1, 4, 4, 4), requires_grad=True)
        target = torch.rand((1, 1, 4, 4, 4))
        voxel_weight = torch.ones_like(target)
        case_weight = torch.ones(1)
        base = protocol_loss(logits, target, voxel_weight, case_weight)
        with_empty_alignment = protocol_loss(
            logits,
            target,
            voxel_weight,
            case_weight,
            alignment_core_mask=torch.zeros_like(target),
            alignment_core_target=torch.zeros_like(target),
            alignment_core_weight=torch.zeros_like(target),
            alignment_teacher=torch.zeros_like(target),
            alignment_trust_weight=torch.zeros_like(target),
            alignment_core_loss_weight=0.08,
            alignment_distill_loss_weight=0.15,
        )
        self.assertTrue(torch.allclose(base, with_empty_alignment))

    def test_alignment_sampler_centers_core_patch(self) -> None:
        shape = (16, 16, 16)
        image = np.zeros(shape, dtype=np.float32)
        target = np.zeros(shape, dtype=np.float32)
        voxel_weight = np.ones(shape, dtype=np.float32)
        alignment_core_mask = np.zeros(shape, dtype=np.float32)
        alignment_core_target = np.zeros(shape, dtype=np.float32)
        alignment_core_weight = np.zeros(shape, dtype=np.float32)
        alignment_core_mask[8, 8, 8] = 1.0
        alignment_core_target[8, 8, 8] = 1.0
        alignment_core_weight[8, 8, 8] = 1.0
        case = PreparedCase(
            case_id="case_a",
            image=image,
            target=target,
            voxel_weight=voxel_weight,
            case_weight=1.0,
            alignment_core_mask=alignment_core_mask,
            alignment_core_target=alignment_core_target,
            alignment_core_weight=alignment_core_weight,
            alignment_sampling_mask=alignment_core_mask,
        )
        dataset = RoundPatchDataset(
            [case],
            patch_size=(4, 4, 4),
            patches_per_case=1,
            positive_patch_probability=0.0,
            seed=123,
            alignment_core_patch_probability=1.0,
        )
        batch = dataset[0]
        self.assertGreater(float(batch["alignment_core_mask"].sum()), 0.0)
        self.assertGreater(float(batch["alignment_core_weight"].sum()), 0.0)

    def test_positive_sampler_falls_back_when_soft_target_has_no_foreground(self) -> None:
        shape = (8, 8, 8)
        case = PreparedCase(
            case_id="case_a",
            image=np.zeros(shape, dtype=np.float32),
            target=np.full(shape, 1.0e-4, dtype=np.float32),
            voxel_weight=np.ones(shape, dtype=np.float32),
            case_weight=1.0,
        )
        dataset = RoundPatchDataset(
            [case],
            patch_size=(4, 4, 4),
            patches_per_case=1,
            positive_patch_probability=1.0,
            seed=123,
        )
        batch = dataset[0]
        self.assertEqual(tuple(batch["image"].shape), (1, 4, 4, 4))


if __name__ == "__main__":
    unittest.main()
