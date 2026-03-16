"""Protocol losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, voxel_weight: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * voxel_weight).mean(dim=(1, 2, 3, 4))


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3, 4))
    denom = probs.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice


def protocol_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    voxel_weight: torch.Tensor,
    case_weight: torch.Tensor,
) -> torch.Tensor:
    wbce = weighted_bce_with_logits(logits, target, voxel_weight)
    sdice = soft_dice_loss(logits, target)
    per_case = case_weight * (0.5 * wbce + 0.5 * sdice)
    return per_case.mean()
