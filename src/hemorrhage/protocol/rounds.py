"""Round-level helpers for targets, stop metrics, and state transitions."""

from __future__ import annotations

from statistics import median
from typing import Any

import numpy as np

from hemorrhage.protocol.metrics import dice_score, edit_ratio, fused_uncertainty_map, mae


def build_soft_target(binary_mask: np.ndarray, oof_probability: np.ndarray, alpha: float) -> np.ndarray:
    target = (1.0 - alpha) * binary_mask.astype(np.float32) + alpha * oof_probability.astype(np.float32)
    return np.clip(target, 1.0e-4, 1.0 - 1.0e-4).astype(np.float32)


def build_voxel_weights(train_uncertainty: np.ndarray, reviewed: bool, floor_weight: float = 0.5) -> np.ndarray:
    if reviewed:
        return np.ones_like(train_uncertainty, dtype=np.float32)
    return np.clip(1.0 - 0.5 * train_uncertainty, floor_weight, 1.0).astype(np.float32)


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
) -> dict[str, float]:
    routine_mask = set(routine_ids)
    audit_mask = set(audit_ids)
    routine_edits: list[float] = []
    audit_edits: list[float] = []
    routine_dice: list[float] = []
    audit_dice: list[float] = []
    delta_t_list: list[float] = []
    all_uncertainty = []
    high_uncertainty = 0
    total_voxels = 0
    coverage = sum(1 for case in cases if int(case["review_count"]) > 0) / max(len(cases), 1)
    anchor_dice_values: list[float] = []

    for case in cases:
        case_id = case["case_id"]
        y_prev = case["previous_binary_label"]
        y_curr = case["current_binary_label"]
        prev_pred_binary = (previous_predictions[case_id] > 0.5).astype(np.uint8)
        delta_t_list.append(mae(current_targets[case_id], previous_targets[case_id]))
        uncertainty = current_uncertainty[case_id]
        all_uncertainty.append(float(uncertainty.mean()))
        high_uncertainty += int((uncertainty > 0.5).sum())
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
        "delta_t": _median_or_zero(delta_t_list) if delta_t_list else 0.0,
        "stab_audit": _median_or_zero(anchor_dice_values),
        "mean_fused_uncertainty": float(np.mean(all_uncertainty)) if all_uncertainty else 0.0,
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
