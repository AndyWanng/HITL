"""Round-level helpers for targets, stop metrics, and state transitions."""

from __future__ import annotations

from statistics import median
from typing import Any

import numpy as np
from scipy import ndimage

from hemorrhage.protocol.metrics import dice_score, edit_ratio, fused_uncertainty_map


def build_soft_target(binary_mask: np.ndarray, oof_probability: np.ndarray, alpha: float) -> np.ndarray:
    target = (1.0 - alpha) * binary_mask.astype(np.float32) + alpha * oof_probability.astype(np.float32)
    return np.clip(target, 1.0e-4, 1.0 - 1.0e-4).astype(np.float32)


def build_voxel_weights(train_uncertainty: np.ndarray, reviewed: bool, floor_weight: float = 0.5) -> np.ndarray:
    if reviewed:
        return np.ones_like(train_uncertainty, dtype=np.float32)
    return np.clip(1.0 - 0.5 * train_uncertainty, floor_weight, 1.0).astype(np.float32)


def build_training_target_payload(
    previous_binary_mask: np.ndarray,
    current_binary_mask: np.ndarray,
    oof_probability: np.ndarray,
    alpha: float,
    reviewed: bool,
    floor_weight: float = 0.5,
    alignment_cfg: dict[str, Any] | None = None,
    teacher_probability: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    cfg = alignment_cfg or {}
    teacher = (
        teacher_probability.astype(np.float32, copy=False)
        if teacher_probability is not None
        else oof_probability.astype(np.float32, copy=False)
    )
    teacher = np.clip(teacher, 1.0e-4, 1.0 - 1.0e-4).astype(np.float32)
    residual_mode = bool(cfg.get("enabled", False)) and str(cfg.get("mode", "")) == "teacher_residual_no_regret"
    unreviewed_target = str(cfg.get("unreviewed_target", "teacher"))
    reviewed_supervision = str(cfg.get("reviewed_supervision", "edit_only"))
    if residual_mode and (
        (not reviewed and unreviewed_target == "teacher")
        or (reviewed and reviewed_supervision == "edit_only")
    ):
        target = teacher.copy()
    else:
        target = build_soft_target(current_binary_mask, oof_probability, alpha)
    uncertainty = fused_uncertainty_map(target, alpha)
    voxel_weight = build_voxel_weights(uncertainty, reviewed=reviewed, floor_weight=floor_weight)
    alignment_core_mask = np.zeros_like(current_binary_mask, dtype=np.float32)
    alignment_core_target = np.zeros_like(current_binary_mask, dtype=np.float32)
    alignment_core_weight = np.zeros_like(current_binary_mask, dtype=np.float32)
    alignment_sampling_mask = np.zeros_like(current_binary_mask, dtype=np.float32)
    alignment_trust_weight = np.zeros_like(current_binary_mask, dtype=np.float32)
    if not bool(cfg.get("enabled", False)):
        return _training_payload(
            target,
            voxel_weight,
            uncertainty,
            alignment_core_mask,
            alignment_core_target,
            alignment_core_weight,
            alignment_sampling_mask,
            teacher,
            alignment_trust_weight,
        )

    previous = previous_binary_mask.astype(bool)
    current = current_binary_mask.astype(bool)
    added = np.logical_and(current, ~previous)
    removed = np.logical_and(previous, ~current)
    oof = oof_probability.astype(np.float32, copy=False)
    added_core = added & (oof < float(cfg.get("added_core_threshold", 0.3)))
    removed_core = removed & (oof > float(cfg.get("removed_core_threshold", 0.5)))
    core_mask = np.logical_or(added_core, removed_core) if reviewed else np.zeros_like(current, dtype=bool)
    if core_mask.any():
        alignment_core_mask[core_mask] = 1.0
        alignment_core_target[added_core] = 1.0
        core_strength = np.abs(current_binary_mask.astype(np.float32) - oof)
        alignment_core_weight[core_mask] = np.maximum(core_strength[core_mask], float(cfg.get("core_weight_floor", 0.25)))
        iterations = max(0, int(cfg.get("core_dilation_iterations", 0)))
        if iterations > 0:
            sampling_mask = ndimage.binary_dilation(core_mask, structure=np.ones((3, 3, 3), dtype=np.uint8), iterations=iterations)
        else:
            sampling_mask = core_mask
        alignment_sampling_mask[sampling_mask.astype(bool)] = 1.0

    confidence = np.abs(teacher - 0.5) * 2.0
    confidence_threshold = float(cfg.get("teacher_confidence_threshold", 0.85))
    trust_region = (confidence >= confidence_threshold) & (alignment_core_mask <= 0.0)
    if bool(cfg.get("trust_use_confidence_weight", True)):
        alignment_trust_weight[trust_region] = confidence[trust_region]
    else:
        alignment_trust_weight[trust_region] = 1.0

    return _training_payload(
        target,
        voxel_weight,
        uncertainty,
        alignment_core_mask,
        alignment_core_target,
        alignment_core_weight,
        alignment_sampling_mask,
        teacher,
        alignment_trust_weight,
    )


def _training_payload(
    target: np.ndarray,
    voxel_weight: np.ndarray,
    uncertainty: np.ndarray,
    alignment_core_mask: np.ndarray,
    alignment_core_target: np.ndarray,
    alignment_core_weight: np.ndarray,
    alignment_sampling_mask: np.ndarray,
    alignment_teacher: np.ndarray,
    alignment_trust_weight: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "target": target.astype(np.float32),
        "voxel_weight": voxel_weight.astype(np.float32),
        "alignment_core_mask": alignment_core_mask.astype(np.float32),
        "alignment_core_target": alignment_core_target.astype(np.float32),
        "alignment_core_weight": alignment_core_weight.astype(np.float32),
        "alignment_sampling_mask": alignment_sampling_mask.astype(np.float32),
        "alignment_teacher": alignment_teacher.astype(np.float32),
        "alignment_trust_weight": alignment_trust_weight.astype(np.float32),
        "uncertainty": uncertainty.astype(np.float32),
    }


def oof_mean_and_variance(tta_predictions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(tta_predictions, axis=0).astype(np.float32)
    return stacked.mean(axis=0), (4.0 * stacked.var(axis=0)).astype(np.float32)


def compute_review_reentry(anchor_dice: float, review_cfg: dict[str, Any], current_round: int, role: str) -> int:
    if role == "routine":
        return current_round + int(review_cfg["routine_reentry_delay"])
    audit_cfg = review_cfg["audit_reentry"]
    if anchor_dice >= float(audit_cfg["high_dice"]):
        return current_round + int(audit_cfg["delay_high"])
    if anchor_dice >= float(audit_cfg["medium_dice"]):
        return current_round + int(audit_cfg["delay_medium"])
    return current_round + int(audit_cfg["delay_low"])


def compute_round_summary(
    cases: list[dict[str, Any]],
    routine_ids: list[str],
    audit_ids: list[str],
    previous_targets: dict[str, np.ndarray],
    current_targets: dict[str, np.ndarray],
    previous_predictions: dict[str, np.ndarray],
    current_uncertainty: dict[str, np.ndarray],
    review_records: dict[str, dict[str, Any]],
    previous_binary_labels: dict[str, np.ndarray] | None = None,
    high_uncertainty_threshold: float = 0.5,
) -> dict[str, float]:
    routine_mask = set(routine_ids)
    audit_mask = set(audit_ids)
    routine_edits: list[float] = []
    audit_edits: list[float] = []
    routine_dice: list[float] = []
    audit_dice: list[float] = []
    delta_abs_sum = 0.0
    delta_voxels = 0
    uncertainty_sum = 0.0
    high_uncertainty = 0
    total_voxels = 0
    coverage = sum(1 for case in cases if int(case["review_count"]) > 0) / max(len(cases), 1)
    anchor_dice_values: list[float] = []

    for case in cases:
        case_id = case["case_id"]
        y_prev = (previous_binary_labels or {}).get(case_id, case["previous_binary_label"])
        y_curr = case["current_binary_label"]
        prev_pred_binary = (previous_predictions[case_id] > 0.5).astype(np.uint8)
        delta = np.abs(current_targets[case_id] - previous_targets[case_id])
        delta_abs_sum += float(delta.sum())
        delta_voxels += int(delta.size)
        uncertainty = current_uncertainty[case_id]
        uncertainty_sum += float(uncertainty.sum())
        high_uncertainty += int((uncertainty > high_uncertainty_threshold).sum())
        total_voxels += uncertainty.size
        if case_id in routine_mask:
            routine_edits.append(edit_ratio(y_curr, y_prev))
            routine_dice.append(dice_score(prev_pred_binary, y_curr))
        if case_id in audit_mask:
            audit_edits.append(edit_ratio(y_curr, y_prev))
            audit_dice.append(dice_score(prev_pred_binary, y_curr))
            record = review_records.get(case_id, {})
            if "anchor_binary" in record and "final_binary" in record:
                anchor_dice_values.append(dice_score(record["anchor_binary"], record["final_binary"]))

    def _median_or_zero(values: list[float]) -> float:
        return float(median(values)) if values else 0.0

    routine_times = [
        float(review_records[cid]["review_time"])
        for cid in routine_ids
        if review_records.get(cid, {}).get("review_time") is not None
    ]
    anchor_times = [
        float(review_records[cid]["anchor_time"])
        for cid in audit_ids
        if review_records.get(cid, {}).get("anchor_time") is not None
    ]
    assisted_times = [
        float(review_records[cid]["assisted_time"])
        for cid in audit_ids
        if review_records.get(cid, {}).get("assisted_time") is not None
    ]
    return {
        "cov": float(coverage),
        "edit_routine": _median_or_zero(routine_edits),
        "edit_audit": _median_or_zero(audit_edits),
        "delta_t": float(delta_abs_sum / max(delta_voxels, 1)),
        "stab_audit": _median_or_zero(anchor_dice_values),
        "mean_fused_uncertainty": float(uncertainty_sum / max(total_voxels, 1)),
        "high_uncertainty_fraction": float(high_uncertainty / max(total_voxels, 1)),
        "dice_model_final_routine": _median_or_zero(routine_dice),
        "dice_model_final_audit": _median_or_zero(audit_dice),
        "routine_median_time": _median_or_zero(routine_times),
        "audit_anchor_median_time": _median_or_zero(anchor_times),
        "audit_assisted_median_time": _median_or_zero(assisted_times),
    }


def compute_stop_state(round_metrics: list[dict[str, float]], coverage: float, breadth_threshold: float, stop_cfg: dict[str, Any]) -> dict[str, Any]:
    patience = int(stop_cfg["patience_non_empty_rounds"])
    recent = round_metrics[-patience:]
    criteria_met = False
    if len(round_metrics) >= patience and coverage >= breadth_threshold:
        criteria_met = all(
            item["edit_routine"] < float(stop_cfg["tau_routine"])
            and item["edit_audit"] < float(stop_cfg["tau_audit"])
            and item["delta_t"] < float(stop_cfg["tau_delta"])
            and item["stab_audit"] > float(stop_cfg["tau_anchor_stability"])
            for item in recent
        )
    return {
        "breadth_threshold": breadth_threshold,
        "non_empty_rounds": len(round_metrics),
        "should_stop": criteria_met,
    }


def compute_uncertainty_from_target(soft_target: np.ndarray, alpha: float) -> np.ndarray:
    return fused_uncertainty_map(soft_target, alpha)
