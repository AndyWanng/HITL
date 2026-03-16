"""Case scoring, budget split, and selection logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hemorrhage.protocol.metrics import connected_components_score, positive_slice_fraction, top_n_mean


@dataclass(slots=True)
class ScoredCase:
    case_id: str
    fold_id: int
    review_count: int
    last_review_round: int
    d: float
    u: float
    c: float
    d_bar: float = 0.0
    u_bar: float = 0.0
    c_bar: float = 0.0
    benefit: float = 0.0
    score: float = 0.0
    score_eff: float = 0.0
    role: str | None = None


def split_budget(total_budget: int) -> tuple[int, int]:
    audit = max(1, round(total_budget / 3))
    routine = total_budget - audit
    return routine, audit


def compute_case_scores(cases: list[dict[str, Any]], protocol_cfg: Any) -> list[ScoredCase]:
    selection_cfg = protocol_cfg.selection
    scored: list[ScoredCase] = []
    top_fraction = float(selection_cfg["top_voxel_fraction"])
    top_min = int(selection_cfg["top_voxel_min"])
    max_components = int(selection_cfg["review_cost"]["max_connected_components"])

    for case in cases:
        y_prev = case["previous_binary_label"]
        s_prev = case["previous_oof"]
        q_prev = case["previous_q"]
        n_voxels = max(top_min, int(np.ceil(y_prev.size * top_fraction)))
        disagreement = top_n_mean(np.abs(y_prev - s_prev), n_voxels)
        uncertainty = top_n_mean(q_prev, n_voxels)
        pred_binary = (s_prev > 0.5).astype(np.uint8)
        positive_fraction = float(pred_binary.mean())
        positive_slices = positive_slice_fraction(pred_binary)
        _, cc_score = connected_components_score(pred_binary, max_components=max_components)
        cost = (
            float(selection_cfg["review_cost"]["base_bias"])
            + float(selection_cfg["review_cost"]["positive_voxel_weight"]) * positive_fraction
            + float(selection_cfg["review_cost"]["positive_slice_weight"]) * positive_slices
            + float(selection_cfg["review_cost"]["connected_component_weight"]) * cc_score
        )
        scored.append(
            ScoredCase(
                case_id=case["case_id"],
                fold_id=int(case["fold_id"]),
                review_count=int(case["review_count"]),
                last_review_round=int(case["last_review_round"]),
                d=disagreement,
                u=uncertainty,
                c=cost,
            )
        )

    _normalize_in_place(scored, "d", "d_bar")
    _normalize_in_place(scored, "u", "u_bar")
    _normalize_in_place(scored, "c", "c_bar")
    for item in scored:
        item.benefit = (
            float(selection_cfg["benefit_disagreement_weight"]) * item.d_bar
            + float(selection_cfg["benefit_uncertainty_weight"]) * item.u_bar
        )
        item.score = item.benefit / (1.0 + item.c_bar)
        item.score_eff = item.score / (1.0 + float(selection_cfg["routine_repeat_penalty"]) * item.review_count)
    return scored


def _normalize_in_place(items: list[ScoredCase], src: str, dst: str) -> None:
    values = [getattr(item, src) for item in items]
    low = min(values)
    high = max(values)
    if abs(high - low) < 1.0e-12:
        for item in items:
            setattr(item, dst, 0.0)
        return
    for item in items:
        setattr(item, dst, (getattr(item, src) - low) / (high - low))


def select_routine(scored: list[ScoredCase], routine_budget: int) -> list[ScoredCase]:
    ordered = sorted(
        scored,
        key=lambda item: (-item.score_eff, item.review_count, item.last_review_round, item.case_id),
    )
    selected = ordered[:routine_budget]
    for item in selected:
        item.role = "routine"
    return selected


def build_audit_pool(scored: list[ScoredCase], routine_ids: set[str], audit_budget: int) -> list[ScoredCase]:
    leftovers = [item for item in scored if item.case_id not in routine_ids]
    first = [item for item in leftovers if item.review_count <= 1]
    if len(first) >= audit_budget:
        return first
    second = [item for item in leftovers if item.review_count <= 2]
    if len(second) >= audit_budget:
        return second
    return leftovers


def select_audit(pool: list[ScoredCase], audit_budget: int, num_folds: int) -> list[ScoredCase]:
    if audit_budget <= 0 or not pool:
        return []
    base_quota = audit_budget // num_folds
    remainder = audit_budget - base_quota * num_folds
    selected: list[ScoredCase] = []
    selected_ids: set[str] = set()

    for fold_id in range(1, num_folds + 1):
        fold_items = [item for item in pool if item.fold_id == fold_id]
        fold_items.sort(key=lambda item: (-item.score, item.case_id))
        selected.extend(_equal_spacing_pick(fold_items, base_quota, selected_ids))

    remaining = [item for item in sorted(pool, key=lambda item: (-item.score, item.case_id)) if item.case_id not in selected_ids]
    selected.extend(_equal_spacing_pick(remaining, remainder, selected_ids))
    for item in selected:
        item.role = "audit"
    return selected


def _equal_spacing_pick(items: list[ScoredCase], count: int, selected_ids: set[str]) -> list[ScoredCase]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        picked = [item for item in items if item.case_id not in selected_ids]
    else:
        picked = []
        for j in range(1, count + 1):
            idx = int(np.ceil(j * len(items) / (count + 1))) - 1
            item = items[idx]
            if item.case_id not in selected_ids:
                picked.append(item)
    for item in picked:
        selected_ids.add(item.case_id)
    return picked
