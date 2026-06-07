"""Protocol losses."""

from __future__ import annotations

import math

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
    alignment_core_mask: torch.Tensor | None = None,
    alignment_core_target: torch.Tensor | None = None,
    alignment_core_weight: torch.Tensor | None = None,
    alignment_teacher: torch.Tensor | None = None,
    alignment_trust_weight: torch.Tensor | None = None,
    alignment_core_loss_weight: float = 0.0,
    alignment_distill_loss_weight: float = 0.0,
    alignment_volume_guard_weight: float = 0.0,
    alignment_added_margin_probability: float = 0.7,
    alignment_removed_margin_probability: float = 0.3,
    alignment_max_volume_ratio: float = 1.03,
    alignment_volume_min_target_mass: float = 1.0,
    alignment_signed_balance_edits: bool = False,
) -> torch.Tensor:
    wbce = weighted_bce_with_logits(logits, target, voxel_weight)
    sdice = soft_dice_loss(logits, target)
    base_loss = 0.5 * wbce + 0.5 * sdice

    if (
        alignment_core_mask is not None
        and alignment_core_target is not None
        and alignment_core_weight is not None
        and alignment_core_loss_weight > 0.0
    ):
        core_weights = alignment_core_mask * alignment_core_weight
        denom = core_weights.sum(dim=(1, 2, 3, 4))
        added_margin = _probability_to_logit(alignment_added_margin_probability, logits)
        removed_margin = _probability_to_logit(alignment_removed_margin_probability, logits)
        added_loss = F.softplus(added_margin - logits)
        removed_loss = F.softplus(logits - removed_margin)
        if alignment_signed_balance_edits:
            added_weights = core_weights * (alignment_core_target > 0.5).to(core_weights.dtype)
            removed_weights = core_weights * (alignment_core_target <= 0.5).to(core_weights.dtype)
            added_denom = added_weights.sum(dim=(1, 2, 3, 4))
            removed_denom = removed_weights.sum(dim=(1, 2, 3, 4))
            added_term = (added_loss * added_weights).sum(dim=(1, 2, 3, 4)) / added_denom.clamp_min(1.0e-6)
            removed_term = (removed_loss * removed_weights).sum(dim=(1, 2, 3, 4)) / removed_denom.clamp_min(1.0e-6)
            active_added = (added_denom > 1.0e-6).to(base_loss.dtype)
            active_removed = (removed_denom > 1.0e-6).to(base_loss.dtype)
            active_terms = (active_added + active_removed).clamp_min(1.0)
            core_loss = (added_term * active_added + removed_term * active_removed) / active_terms
        else:
            margin_loss = torch.where(alignment_core_target > 0.5, added_loss, removed_loss)
            core_loss = (margin_loss * core_weights).sum(dim=(1, 2, 3, 4)) / denom.clamp_min(1.0e-6)
        active = (denom > 1.0e-6).to(base_loss.dtype)
        base_loss = base_loss + float(alignment_core_loss_weight) * core_loss * active

    if (
        alignment_teacher is not None
        and alignment_trust_weight is not None
        and alignment_distill_loss_weight > 0.0
    ):
        trust_denom = alignment_trust_weight.sum(dim=(1, 2, 3, 4))
        distill_bce = F.binary_cross_entropy_with_logits(logits, alignment_teacher, reduction="none")
        distill_loss = (distill_bce * alignment_trust_weight).sum(dim=(1, 2, 3, 4)) / trust_denom.clamp_min(1.0e-6)
        active = (trust_denom > 1.0e-6).to(base_loss.dtype)
        base_loss = base_loss + float(alignment_distill_loss_weight) * distill_loss * active

    if alignment_volume_guard_weight > 0.0:
        probs = torch.sigmoid(logits)
        pred_mass = probs.sum(dim=(1, 2, 3, 4))
        target_mass = target.sum(dim=(1, 2, 3, 4))
        active = (target_mass > float(alignment_volume_min_target_mass)).to(base_loss.dtype)
        volume_ratio = pred_mass / target_mass.clamp_min(float(alignment_volume_min_target_mass))
        volume_loss = torch.relu(volume_ratio - float(alignment_max_volume_ratio)).pow(2)
        base_loss = base_loss + float(alignment_volume_guard_weight) * volume_loss * active

    per_case = case_weight * base_loss
    return per_case.mean()


def _probability_to_logit(probability: float, reference: torch.Tensor) -> torch.Tensor:
    clipped = min(max(float(probability), 1.0e-4), 1.0 - 1.0e-4)
    value = math.log(clipped / (1.0 - clipped))
    return reference.new_tensor(value)
