"""Safe gated revision policy for round-level HITL correction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hemorrhage.protocol.metrics import dice_score
from hemorrhage.utils import ensure_dir


EPS = 1.0e-6


@dataclass(slots=True)
class RevisionEditMaps:
    review_added: np.ndarray
    review_removed: np.ndarray
    base_added: np.ndarray
    base_removed: np.ndarray


@dataclass(slots=True)
class RevisionCase:
    case_id: str
    fold_id: int
    role: str | None
    image: np.ndarray
    p_round0: np.ndarray
    q_round0: np.ndarray
    p_base: np.ndarray
    q_base: np.ndarray
    y_old: np.ndarray
    y_final: np.ndarray
    edit_maps: RevisionEditMaps
    action_region_eval: np.ndarray
    action_region_train: np.ndarray


@dataclass(slots=True)
class ComponentCandidate:
    case_id: str
    action: str
    mask: np.ndarray
    features: np.ndarray
    oracle_gain: float | None = None
    oracle_label: int | None = None
    probability: float | None = None
    predicted_gain: float | None = None


def _as_bool(array: np.ndarray) -> np.ndarray:
    return np.asarray(array).astype(bool)


def compute_revision_edit_maps(
    y_old: np.ndarray,
    y_final: np.ndarray,
    p_base: np.ndarray,
    threshold: float = 0.5,
) -> RevisionEditMaps:
    """Return review-edit and model-correction edit maps.

    SGRA supervision uses base_added/base_removed because it corrects the frozen
    base prediction, while review_added/review_removed are retained for reports.
    """

    old = _as_bool(y_old)
    final = _as_bool(y_final)
    base = np.asarray(p_base) >= float(threshold)
    return RevisionEditMaps(
        review_added=np.logical_and(final, ~old),
        review_removed=np.logical_and(old, ~final),
        base_added=np.logical_and(final, ~base),
        base_removed=np.logical_and(base, ~final),
    )


def _binary_structure() -> np.ndarray:
    return np.ones((3, 3, 3), dtype=bool)


def boundary_band(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    mask = _as_bool(mask)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    iterations = max(1, int(iterations))
    dilated = ndimage.binary_dilation(mask, structure=_binary_structure(), iterations=iterations)
    eroded = ndimage.binary_erosion(mask, structure=_binary_structure(), iterations=iterations, border_value=0)
    return np.logical_and(dilated, ~eroded)


def _high_quantile_mask(values: np.ndarray, quantile: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or not np.isfinite(values).any() or float(values.max()) <= 0.0:
        return np.zeros_like(values, dtype=bool)
    threshold = float(np.quantile(values[np.isfinite(values)], np.clip(float(quantile), 0.0, 1.0)))
    if threshold <= 0.0:
        return values > 0.0
    return values >= threshold


def make_action_region(
    p_round0: np.ndarray,
    p_base: np.ndarray,
    q_round0: np.ndarray,
    q_base: np.ndarray,
    edit_maps: RevisionEditMaps | None,
    candidate_cfg: dict[str, Any],
    include_training_edits: bool,
) -> np.ndarray:
    base_thr = float(candidate_cfg.get("p_base_threshold", 0.5))
    p0_thr = float(candidate_cfg.get("p0_threshold", 0.3))
    low_thr = float(candidate_cfg.get("low_threshold", 0.25))
    disagreement_thr = float(candidate_cfg.get("disagreement_threshold", 0.25))
    boundary_iterations = int(candidate_cfg.get("boundary_dilation_iterations", 2))
    edit_iterations = int(candidate_cfg.get("edit_dilation_iterations", 2))
    m_base = np.asarray(p_base) >= base_thr
    m0_hard = np.asarray(p_round0) >= base_thr
    region = np.zeros_like(m_base, dtype=bool)
    region |= boundary_band(m_base, boundary_iterations)
    region |= np.logical_xor(m0_hard, m_base)
    region |= np.abs(np.asarray(p_base, dtype=np.float32) - np.asarray(p_round0, dtype=np.float32)) > disagreement_thr
    region |= _high_quantile_mask(q_base, float(candidate_cfg.get("teacher_uncertainty_quantile", 0.995)))
    region |= _high_quantile_mask(q_round0, float(candidate_cfg.get("round0_uncertainty_quantile", 0.995)))
    region |= np.logical_and(np.asarray(p_base) >= low_thr, np.asarray(p_base) <= (1.0 - low_thr))
    region |= np.asarray(p_round0) >= p0_thr
    if include_training_edits and edit_maps is not None:
        edit = np.logical_or(edit_maps.base_added, edit_maps.base_removed)
        if edit.any():
            region |= ndimage.binary_dilation(edit, structure=_binary_structure(), iterations=max(0, edit_iterations))
    return region.astype(bool)


def make_revision_case(
    *,
    case_id: str,
    fold_id: int,
    role: str | None,
    image: np.ndarray,
    p_round0: np.ndarray,
    q_round0: np.ndarray,
    p_base: np.ndarray,
    q_base: np.ndarray,
    y_old: np.ndarray,
    y_final: np.ndarray,
    candidate_cfg: dict[str, Any],
) -> RevisionCase:
    edit_maps = compute_revision_edit_maps(
        y_old=y_old,
        y_final=y_final,
        p_base=p_base,
        threshold=float(candidate_cfg.get("p_base_threshold", 0.5)),
    )
    action_eval = make_action_region(p_round0, p_base, q_round0, q_base, edit_maps, candidate_cfg, include_training_edits=False)
    action_train = make_action_region(p_round0, p_base, q_round0, q_base, edit_maps, candidate_cfg, include_training_edits=role is not None)
    return RevisionCase(
        case_id=case_id,
        fold_id=fold_id,
        role=role,
        image=image.astype(np.float32, copy=False),
        p_round0=p_round0.astype(np.float32, copy=False),
        q_round0=q_round0.astype(np.float32, copy=False),
        p_base=p_base.astype(np.float32, copy=False),
        q_base=q_base.astype(np.float32, copy=False),
        y_old=y_old.astype(np.uint8, copy=False),
        y_final=y_final.astype(np.uint8, copy=False),
        edit_maps=edit_maps,
        action_region_eval=action_eval,
        action_region_train=action_train,
    )


def _component_slices(mask: np.ndarray, min_component_voxels: int) -> list[np.ndarray]:
    labels, count = ndimage.label(_as_bool(mask), structure=_binary_structure())
    components: list[np.ndarray] = []
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        if int(comp.sum()) >= int(min_component_voxels):
            components.append(comp)
    return components


def _component_slice_span(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0
    return int(coords[:, 2].max() - coords[:, 2].min() + 1)


def _safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(np.asarray(values)[mask].mean())


def _safe_max(values: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(np.asarray(values)[mask].max())


def _feature_vector(case: RevisionCase, mask: np.ndarray, action: str) -> np.ndarray:
    base_thr = 0.5
    base_mask = case.p_base >= base_thr
    round0_mask = case.p_round0 >= base_thr
    old = case.y_old.astype(bool)
    diff = np.abs(case.p_base - case.p_round0)
    boundary = boundary_band(base_mask, 2)
    volume = float(mask.sum())
    denom = max(volume, 1.0)
    features = np.asarray(
        [
            1.0 if action == "add" else 0.0,
            math.log1p(volume),
            float(_component_slice_span(mask)),
            _safe_mean(case.p_base, mask),
            _safe_max(case.p_base, mask),
            _safe_mean(case.p_round0, mask),
            _safe_max(case.p_round0, mask),
            _safe_mean(diff, mask),
            _safe_mean(case.q_base, mask),
            _safe_max(case.q_base, mask),
            _safe_mean(case.q_round0, mask),
            _safe_max(case.q_round0, mask),
            float(np.logical_and(mask, round0_mask).sum() / denom),
            float(np.logical_and(mask, old).sum() / denom),
            float(np.logical_and(mask, np.logical_xor(base_mask, round0_mask)).sum() / denom),
            float(np.logical_and(mask, boundary).sum() / denom),
        ],
        dtype=np.float32,
    )
    return features


def candidate_components(case: RevisionCase, candidate_cfg: dict[str, Any]) -> list[ComponentCandidate]:
    min_voxels = int(candidate_cfg.get("min_component_voxels", 2))
    base_thr = float(candidate_cfg.get("p_base_threshold", 0.5))
    p0_thr = float(candidate_cfg.get("p0_threshold", 0.3))
    low_thr = float(candidate_cfg.get("low_threshold", 0.25))
    base_mask = case.p_base >= base_thr
    add_source = np.logical_or(case.p_round0 >= p0_thr, case.action_region_eval)
    add_source &= ~base_mask
    remove_source = base_mask
    candidates: list[ComponentCandidate] = []
    for comp in _component_slices(remove_source, min_voxels):
        candidates.append(ComponentCandidate(case.case_id, "remove", comp, _feature_vector(case, comp, "remove")))
    for comp in _component_slices(np.logical_or(add_source, np.logical_and(case.p_base >= low_thr, case.p_base < base_thr)), min_voxels):
        candidates.append(ComponentCandidate(case.case_id, "add", comp, _feature_vector(case, comp, "add")))
    return candidates


def _apply_component_to_mask(base_mask: np.ndarray, candidate: ComponentCandidate) -> np.ndarray:
    out = base_mask.copy()
    if candidate.action == "remove":
        out[candidate.mask] = False
    else:
        out[candidate.mask] = True
    return out


def label_component_candidates(case: RevisionCase, candidate_cfg: dict[str, Any], abstain_margin: float) -> list[ComponentCandidate]:
    base_mask = case.p_base >= float(candidate_cfg.get("p_base_threshold", 0.5))
    target = case.y_final.astype(bool)
    base_dice = dice_score(base_mask, target)
    labeled: list[ComponentCandidate] = []
    for candidate in candidate_components(case, candidate_cfg):
        new_mask = _apply_component_to_mask(base_mask, candidate)
        gain = float(dice_score(new_mask, target) - base_dice)
        candidate.oracle_gain = gain
        candidate.oracle_label = int(gain > float(abstain_margin))
        labeled.append(candidate)
    return labeled


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def train_component_selector(
    train_cases: list[RevisionCase],
    candidate_cfg: dict[str, Any],
    selector_cfg: dict[str, Any],
) -> dict[str, Any]:
    if str(selector_cfg.get("model", "logistic_ridge")) != "logistic_ridge":
        raise RuntimeError(f"Unsupported revision component selector model: {selector_cfg.get('model')}")
    reviewed_cases = [case for case in train_cases if case.role is not None]
    rows: list[np.ndarray] = []
    labels: list[int] = []
    gains: list[float] = []
    abstain_margin = float(selector_cfg.get("abstain_margin", 0.00025))
    for case in reviewed_cases:
        for candidate in label_component_candidates(case, candidate_cfg, abstain_margin):
            rows.append(candidate.features)
            labels.append(int(candidate.oracle_label or 0))
            gains.append(float(candidate.oracle_gain or 0.0))
    if not rows:
        return {
            "kind": "logistic_ridge",
            "constant_probability": 0.0,
            "num_candidates": 0,
            "num_positive": 0,
            "expected_positive_gain": 0.0,
        }
    x = np.stack(rows).astype(np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1.0e-6] = 1.0
    xs = (x - mean) / std
    positive_gain = [gain for gain, label in zip(gains, labels, strict=True) if label]
    expected_positive_gain = float(np.mean(positive_gain)) if positive_gain else 0.0
    if len(set(labels)) < 2:
        return {
            "kind": "logistic_ridge",
            "constant_probability": float(y.mean()),
            "num_candidates": int(len(labels)),
            "num_positive": int(y.sum()),
            "expected_positive_gain": expected_positive_gain,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
        }
    weights = np.zeros(xs.shape[1], dtype=np.float64)
    bias = float(np.log((y.mean() + 1.0e-3) / (1.0 - y.mean() + 1.0e-3)))
    lr = 0.2
    l2 = float(selector_cfg.get("l2", 1.0))
    for _ in range(300):
        pred = _sigmoid(xs @ weights + bias)
        error = pred - y
        weights -= lr * ((xs.T @ error) / len(y) + l2 * weights / len(y))
        bias -= lr * float(error.mean())
    return {
        "kind": "logistic_ridge",
        "weights": weights.tolist(),
        "bias": bias,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "num_candidates": int(len(labels)),
        "num_positive": int(y.sum()),
        "expected_positive_gain": expected_positive_gain,
    }


def selector_probability(model: dict[str, Any], features: np.ndarray) -> float:
    if "constant_probability" in model:
        return float(model["constant_probability"])
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = float(model["bias"])
    xs = (features.astype(np.float64) - mean) / std
    return float(_sigmoid(np.asarray([xs @ weights + bias]))[0])


def apply_component_selector(
    case: RevisionCase,
    selector_model: dict[str, Any],
    candidate_cfg: dict[str, Any],
    selector_cfg: dict[str, Any],
    probability: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = probability.copy()
    records: list[dict[str, Any]] = []
    threshold = float(selector_cfg.get("apply_threshold", 0.65))
    min_gain = float(selector_cfg.get("min_predicted_gain", 0.0005))
    expected_gain = float(selector_model.get("expected_positive_gain", 0.0))
    for candidate in candidate_components(case, candidate_cfg):
        prob = selector_probability(selector_model, candidate.features)
        predicted_gain = max(0.0, (prob - 0.5) * 2.0 * expected_gain)
        candidate.probability = prob
        candidate.predicted_gain = predicted_gain
        applied = prob >= threshold and predicted_gain >= min_gain
        if applied:
            if candidate.action == "remove":
                out[candidate.mask] = np.minimum(out[candidate.mask], 0.05)
            else:
                out[candidate.mask] = np.maximum(out[candidate.mask], np.maximum(case.p_round0[candidate.mask], 0.75))
        records.append(
            {
                "case_id": case.case_id,
                "action": candidate.action,
                "voxels": int(candidate.mask.sum()),
                "probability": prob,
                "predicted_gain": predicted_gain,
                "applied": applied,
            }
        )
    return np.clip(out, 1.0e-4, 1.0 - 1.0e-4).astype(np.float32), records


class SGRAAdapterNet(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(12, 12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(12, 4, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    mask = _as_bool(mask)
    outside = ndimage.distance_transform_edt(~mask)
    inside = ndimage.distance_transform_edt(mask)
    dist = outside - inside
    scale = max(float(np.percentile(np.abs(dist), 95)), 1.0)
    return np.clip(dist / scale, -1.0, 1.0).astype(np.float32)


def adapter_input(case: RevisionCase, channels: list[str]) -> np.ndarray:
    base_mask = case.p_base >= 0.5
    values: dict[str, np.ndarray] = {
        "image": case.image.astype(np.float32),
        "p_base": case.p_base.astype(np.float32),
        "p_round0": case.p_round0.astype(np.float32),
        "abs_base_round0": np.abs(case.p_base - case.p_round0).astype(np.float32),
        "q_base": case.q_base.astype(np.float32),
        "q_round0": case.q_round0.astype(np.float32),
        "boundary_distance": _signed_distance(base_mask),
        "old_label": case.y_old.astype(np.float32),
    }
    return np.stack([values[name] for name in channels], axis=0).astype(np.float32)


class RevisionPatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        cases: list[RevisionCase],
        channels: list[str],
        patch_size: tuple[int, int, int],
        patches_per_case: int,
        adapter_cfg: dict[str, Any],
        seed: int,
    ) -> None:
        self.cases = cases
        self.channels = channels
        self.patch_size = np.asarray(patch_size, dtype=np.int32)
        self.patches_per_case = int(patches_per_case)
        self.rng = np.random.default_rng(seed)
        self.probs = np.asarray(
            [
                float(adapter_cfg.get("add_patch_probability", 0.3)),
                float(adapter_cfg.get("remove_patch_probability", 0.3)),
                float(adapter_cfg.get("hard_negative_patch_probability", 0.2)),
                float(adapter_cfg.get("identity_patch_probability", 0.2)),
            ],
            dtype=np.float64,
        )
        if self.probs.sum() <= 0:
            self.probs[:] = 0.25
        self.probs /= self.probs.sum()

    def __len__(self) -> int:
        return max(1, len(self.cases) * self.patches_per_case)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case = self.cases[index % len(self.cases)]
        center = self._choose_center(case)
        slices = self._slices(case.image.shape, center)
        base_mask = case.p_base >= 0.5
        add_target = case.edit_maps.base_added.astype(np.float32)
        remove_target = case.edit_maps.base_removed.astype(np.float32)
        action = case.action_region_train.astype(np.float32)
        reviewed = float(case.role is not None)
        identity = np.logical_or(~case.action_region_train, np.logical_or(case.p_base < 0.05, case.p_base > 0.95)).astype(np.float32)
        x = adapter_input(case, self.channels)
        return {
            "input": torch.from_numpy(x[(slice(None),) + slices]),
            "p_base": torch.from_numpy(case.p_base[slices][None].astype(np.float32)),
            "base_mask": torch.from_numpy(base_mask[slices][None].astype(np.float32)),
            "add_target": torch.from_numpy(add_target[slices][None]),
            "remove_target": torch.from_numpy(remove_target[slices][None]),
            "action": torch.from_numpy(action[slices][None]),
            "identity": torch.from_numpy(identity[slices][None]),
            "reviewed": torch.tensor(reviewed, dtype=torch.float32),
        }

    def _choose_center(self, case: RevisionCase) -> np.ndarray:
        draw = self.rng.choice(4, p=self.probs)
        masks = [
            case.edit_maps.base_added,
            case.edit_maps.base_removed,
            np.logical_and(case.action_region_train, ~(case.edit_maps.base_added | case.edit_maps.base_removed)),
            np.ones_like(case.action_region_train, dtype=bool),
        ]
        for mask in [masks[draw], *masks]:
            coords = np.argwhere(mask)
            if coords.size:
                return coords[self.rng.integers(len(coords))]
        return np.asarray(case.image.shape) // 2

    def _slices(self, shape: tuple[int, int, int], center: np.ndarray) -> tuple[slice, slice, slice]:
        starts = np.asarray(center, dtype=np.int32) - self.patch_size // 2
        starts = np.clip(starts, 0, np.maximum(np.asarray(shape, dtype=np.int32) - self.patch_size, 0))
        ends = starts + self.patch_size
        return tuple(slice(int(starts[idx]), int(ends[idx])) for idx in range(3))


def _balanced_bce(prob: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-6
    prob = torch.clamp(prob, eps, 1.0 - eps)
    pos = mask * target
    neg = mask * (1.0 - target)
    pos_loss = -(pos * torch.log(prob)).sum() / pos.sum().clamp_min(1.0)
    neg_loss = -(neg * torch.log(1.0 - prob)).sum() / neg.sum().clamp_min(1.0)
    return 0.5 * (pos_loss + neg_loss)


def _tv_loss(tensor: torch.Tensor) -> torch.Tensor:
    return (
        torch.mean(torch.abs(tensor[..., 1:, :, :] - tensor[..., :-1, :, :]))
        + torch.mean(torch.abs(tensor[..., :, 1:, :] - tensor[..., :, :-1, :]))
        + torch.mean(torch.abs(tensor[..., :, :, 1:] - tensor[..., :, :, :-1]))
    )


def adapter_forward_probability(
    logits: torch.Tensor,
    p_base: torch.Tensor,
    base_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    add_score = torch.sigmoid(logits[:, 0:1])
    remove_score = torch.sigmoid(logits[:, 1:2])
    gate_add = torch.sigmoid(logits[:, 2:3])
    gate_remove = torch.sigmoid(logits[:, 3:4])
    add_effect = gate_add * add_score * (1.0 - base_mask)
    remove_effect = gate_remove * remove_score * base_mask
    p_new = p_base * (1.0 - remove_effect) + (1.0 - p_base) * add_effect
    return torch.clamp(p_new, 1.0e-4, 1.0 - 1.0e-4), add_score, remove_score, gate_add, gate_remove


def train_adapter(
    train_cases: list[RevisionCase],
    adapter_cfg: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    channels = [str(name) for name in adapter_cfg.get("input_channels", ["image", "p_base", "p_round0", "abs_base_round0"])]
    model = SGRAAdapterNet(len(channels)).to(device)
    patch_size = tuple(int(v) for v in adapter_cfg.get("patch_size", (128, 96, 64)))
    dataset = RevisionPatchDataset(
        train_cases,
        channels=channels,
        patch_size=patch_size,
        patches_per_case=int(adapter_cfg.get("patches_per_case", 4)),
        adapter_cfg=adapter_cfg,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=int(adapter_cfg.get("batch_size", 2)), shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(adapter_cfg.get("lr", 5.0e-5)), weight_decay=1.0e-5)
    history: list[float] = []
    identity_weight = float(adapter_cfg.get("identity_loss_weight", 10.0))
    gate_l1_weight = float(adapter_cfg.get("gate_l1_weight", 0.1))
    gate_tv_weight = float(adapter_cfg.get("gate_tv_weight", 0.05))
    volume_guard_weight = float(adapter_cfg.get("volume_guard_weight", 0.05))
    for _ in range(int(adapter_cfg.get("epochs", 30))):
        losses = []
        model.train()
        for batch in loader:
            inputs = batch["input"].to(device)
            p_base = batch["p_base"].to(device)
            base_mask = batch["base_mask"].to(device)
            action = batch["action"].to(device)
            reviewed = batch["reviewed"].to(device).view(-1, 1, 1, 1, 1)
            supervised = action * reviewed
            identity = batch["identity"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            p_new, add_score, remove_score, gate_add, gate_remove = adapter_forward_probability(logits, p_base, base_mask)
            add_loss = _balanced_bce(add_score, batch["add_target"].to(device), supervised * (1.0 - base_mask))
            remove_loss = _balanced_bce(remove_score, batch["remove_target"].to(device), supervised * base_mask)
            identity_loss = (((p_new - p_base) ** 2) * identity).sum() / identity.sum().clamp_min(1.0)
            gate = action * (gate_add + gate_remove)
            gate_l1 = gate.sum() / action.sum().clamp_min(1.0)
            tv = _tv_loss(gate_add) + _tv_loss(gate_remove)
            volume_delta = torch.abs(p_new.sum(dim=(2, 3, 4)) - p_base.sum(dim=(2, 3, 4))) / p_base.sum(dim=(2, 3, 4)).clamp_min(1.0)
            volume_loss = torch.relu(volume_delta - 0.05).pow(2).mean()
            loss = add_loss + remove_loss + identity_weight * identity_loss + gate_l1_weight * gate_l1 + gate_tv_weight * tv + volume_guard_weight * volume_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else 0.0)
    ensure_dir(checkpoint_path.parent)
    torch.save({"model": model.state_dict(), "channels": channels, "loss_history": history}, checkpoint_path)
    return {"checkpoint_path": str(checkpoint_path), "channels": channels, "loss_history": history}


def predict_adapter(case: RevisionCase, checkpoint_path: Path, adapter_cfg: dict[str, Any], device: torch.device) -> np.ndarray:
    payload = torch.load(checkpoint_path, map_location="cpu")
    channels = [str(name) for name in payload.get("channels", adapter_cfg.get("input_channels", []))]
    model = SGRAAdapterNet(len(channels)).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(adapter_input(case, channels)[None]).to(device)
        p_base = torch.from_numpy(case.p_base[None, None].astype(np.float32)).to(device)
        base_mask = torch.from_numpy((case.p_base >= 0.5)[None, None].astype(np.float32)).to(device)
        logits = model(inputs)
        p_new, _, _, _, _ = adapter_forward_probability(logits, p_base, base_mask)
    out = p_new.detach().cpu().numpy()[0, 0].astype(np.float32)
    out[~case.action_region_eval] = case.p_base[~case.action_region_eval]
    return np.clip(out, 1.0e-4, 1.0 - 1.0e-4).astype(np.float32)


def apply_case_guard(
    p_candidate: np.ndarray,
    p_base: np.ndarray,
    accept_cfg: dict[str, Any],
    threshold: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    base_mask = np.asarray(p_base) >= threshold
    cand_mask = np.asarray(p_candidate) >= threshold
    base_vol = int(base_mask.sum())
    cand_vol = int(cand_mask.sum())
    changed = int(np.logical_xor(base_mask, cand_mask).sum())
    volume_drift = abs(cand_vol - base_vol) / max(base_vol, 1)
    changed_fraction = changed / max(base_vol, 1)
    tiny = base_vol <= int(accept_cfg.get("tiny_lesion_voxels", 100))
    accepted = True
    if tiny:
        accepted = changed <= int(accept_cfg.get("tiny_absolute_changed_voxels", 8))
    else:
        accepted = (
            volume_drift <= float(accept_cfg.get("max_case_volume_drift_fraction", 0.05))
            and changed_fraction <= float(accept_cfg.get("max_changed_fraction_of_base_positive", 0.10))
        )
    return (p_candidate if accepted else p_base).astype(np.float32), {
        "accepted": bool(accepted),
        "base_volume": base_vol,
        "candidate_volume": cand_vol,
        "changed_voxels": changed,
        "volume_drift_fraction": float(volume_drift),
        "changed_fraction_of_base_positive": float(changed_fraction),
        "tiny_lesion": bool(tiny),
    }


def oracle_case_summary(case: RevisionCase, candidate_cfg: dict[str, Any]) -> dict[str, Any]:
    base_mask = case.p_base >= float(candidate_cfg.get("p_base_threshold", 0.5))
    target = case.y_final.astype(bool)
    action = case.action_region_eval
    oracle_mask = base_mask.copy()
    oracle_mask[action] = target[action]
    base_dice = dice_score(base_mask, target)
    oracle_dice = dice_score(oracle_mask, target)
    add = case.edit_maps.base_added
    rem = case.edit_maps.base_removed
    review_add = case.edit_maps.review_added
    review_rem = case.edit_maps.review_removed
    comp_mask = base_mask.copy()
    positive_components = 0
    for candidate in label_component_candidates(case, candidate_cfg, abstain_margin=0.0):
        if float(candidate.oracle_gain or 0.0) > 0.0:
            comp_mask = _apply_component_to_mask(comp_mask, candidate)
            positive_components += 1
    return {
        "case_id": case.case_id,
        "fold_id": case.fold_id,
        "role": case.role or "unreviewed",
        "base_dice": base_dice,
        "candidate_only_oracle_dice": oracle_dice,
        "candidate_only_oracle_gain": float(oracle_dice - base_dice),
        "component_oracle_dice": dice_score(comp_mask, target),
        "component_oracle_gain": float(dice_score(comp_mask, target) - base_dice),
        "positive_oracle_components": int(positive_components),
        "base_added_voxels": int(add.sum()),
        "base_removed_voxels": int(rem.sum()),
        "review_added_voxels": int(review_add.sum()),
        "review_removed_voxels": int(review_rem.sum()),
        "base_add_coverage": float(np.logical_and(add, action).sum() / max(int(add.sum()), 1)),
        "base_remove_coverage": float(np.logical_and(rem, action).sum() / max(int(rem.sum()), 1)),
        "review_add_coverage": float(np.logical_and(review_add, action).sum() / max(int(review_add.sum()), 1)),
        "review_remove_coverage": float(np.logical_and(review_rem, action).sum() / max(int(review_rem.sum()), 1)),
        "action_region_voxels": int(action.sum()),
        "action_region_fraction": float(action.mean()),
    }


def write_selector_model(path: Path, model: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
