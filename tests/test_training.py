from __future__ import annotations

import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()

