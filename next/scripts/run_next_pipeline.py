from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
import torch
import yaml
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANIMALWISE_SCRIPT = PROJECT_ROOT / "analysis" / "animalwise_oof_pipeline" / "scripts" / "run_animalwise_oof.py"


def load_animalwise_module() -> Any:
    spec = importlib.util.spec_from_file_location("animalwise_oof_runtime", str(ANIMALWISE_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load animal-wise runtime module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aw = load_animalwise_module()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value: Any, base: Path = PROJECT_ROOT) -> Optional[Path]:
    if value is None:
        return None
    path = Path(os.path.expandvars(str(value)))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Optional[list[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class TeeLogger:
    def __init__(self, path: Path) -> None:
        ensure_dir(path.parent)
        self.path = path
        self.handle = path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"{timestamp()} {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg.setdefault("paths", {})
    cfg.setdefault("limits", {})
    cfg.setdefault("split", {})
    cfg.setdefault("selection", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("inference", {})
    cfg.setdefault("stages", {})
    cfg.setdefault("outputs", {})
    return cfg


def git_hash() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def command_string() -> str:
    return " ".join([sys.executable, *sys.argv])


def update_state(run_dir: Path, **updates: Any) -> None:
    path = run_dir / "run_state.json"
    state: dict[str, Any] = {}
    if path.exists():
        state = read_json(path)
    state.update(updates)
    state["updated_at"] = timestamp()
    write_json(path, state)


def append_completed_stage(run_dir: Path, stage: str) -> None:
    state_path = run_dir / "run_state.json"
    state = read_json(state_path) if state_path.exists() else {}
    completed = list(state.get("completed_stages", []))
    if stage not in completed:
        completed.append(stage)
    update_state(run_dir, completed_stages=completed)


def ensure_run_dirs(run_dir: Path) -> None:
    for rel in [
        "config",
        "metadata",
        "splits",
        "source_predictions/aramra",
        "scores",
        "selection",
        "checkpoints",
        "predictions/aramra_oof",
        "predictions/epibios_retention",
        "metrics",
        "logs",
        "reports",
    ]:
        ensure_dir(run_dir / rel)


def make_run_dir(cfg: dict[str, Any], explicit_run_dir: Optional[str], resume: Optional[str]) -> Path:
    if resume:
        return resolve_path(resume)  # type: ignore[return-value]
    if explicit_run_dir:
        return resolve_path(explicit_run_dir)  # type: ignore[return-value]
    root = resolve_path(cfg["paths"].get("run_root", "next/runs"))
    if root is None:
        raise ValueError("paths.run_root is required")
    name = cfg.get("experiment_name") or "next_aramra_selection"
    return root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"


def launch_background(args: argparse.Namespace, config_path: Path, cfg: dict[str, Any]) -> None:
    run_dir = make_run_dir(cfg, args.run_dir, args.resume)
    ensure_run_dirs(run_dir)
    resolved_config = run_dir / "config" / "config_resolved.yaml"
    with resolved_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    stdout_path = run_dir / "logs" / "launcher.stdout.log"
    stderr_path = run_dir / "logs" / "launcher.stderr.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path.resolve()),
        "--worker",
        "--run-dir",
        str(run_dir),
    ]
    if args.resume:
        cmd.extend(["--resume", str(args.resume)])
    write_json(
        run_dir / "run_state.json",
        {
            "status": "launching",
            "current_stage": "launcher",
            "started_at": timestamp(),
            "updated_at": timestamp(),
            "pid": None,
            "command": " ".join(cmd),
            "config": str(config_path.resolve()),
            "git_hash": git_hash(),
            "completed_stages": [],
            "failed_stage": None,
        },
    )
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=stdout,
            stderr=stderr,
            start_new_session=os.name != "nt",
            creationflags=creationflags,
        )
    update_state(run_dir, status="running", pid=process.pid)
    print(f"Started background next pipeline run: {run_dir}")
    print(f"PID: {process.pid}")
    print(f"Logs: {run_dir / 'logs'}")


def load_source_fold_map(source_workspace: Path) -> dict[str, int]:
    path = source_workspace / "reports" / "round_1" / "oof_case_metrics.csv"
    if not path.exists():
        path = source_workspace / "reports" / "round_0" / "oof_case_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot locate source fold metrics CSV under {source_workspace}")
    out: dict[str, int] = {}
    for row in read_csv(path):
        out[str(row["case_id"])] = int(float(row["fold_id"]))
    return out


def assign_source_folds(epi_cases: list[Any], source_workspace: Path) -> None:
    fold_map = load_source_fold_map(source_workspace)
    missing = []
    for record in epi_cases:
        if record.case_id not in fold_map:
            missing.append(record.case_id)
        else:
            record.fold = fold_map[record.case_id]
    if missing:
        raise RuntimeError(f"Missing source fold assignments for {len(missing)} EpiBios cases")


def aramra_positive_voxels(record: Any) -> int:
    if record.eval_label_path is None:
        raise ValueError(f"Missing ARAMRA evaluation label path for {record.case_id}")
    return int(aw.load_binary_label(record.eval_label_path).sum())


def make_aramra_folds(records: list[Any], folds: int, seed: int) -> dict[str, int]:
    animals = sorted({record.animal_id_strict for record in records})
    animal_features: dict[str, dict[str, Any]] = {}
    for animal in animals:
        items = [record for record in records if record.animal_id_strict == animal]
        animal_features[animal] = {
            "cases": len(items),
            "pos": sum(aramra_positive_voxels(record) for record in items),
            "time": Counter(record.time_raw for record in items),
            "pattern": Counter(record.timepoint_pattern for record in items),
        }
    totals = {
        "animals": len(animals),
        "cases": len(records),
        "pos": sum(features["pos"] for features in animal_features.values()),
        "time": Counter(),
        "pattern": Counter(),
    }
    for features in animal_features.values():
        totals["time"].update(features["time"])
        totals["pattern"].update(features["pattern"])
    time_keys = sorted(totals["time"])
    pattern_keys = sorted(totals["pattern"])
    stats = [
        {"animals": 0, "cases": 0, "pos": 0, "time": Counter(), "pattern": Counter()}
        for _ in range(folds)
    ]
    rng = np.random.default_rng(seed)
    ordered = sorted(
        animals,
        key=lambda animal: (-animal_features[animal]["pos"], -animal_features[animal]["cases"], rng.random()),
    )
    assignment: dict[str, int] = {}
    for animal in ordered:
        best = min(
            range(folds),
            key=lambda idx: aramra_fold_score(stats, totals, animal_features[animal], idx, folds, time_keys, pattern_keys),
        )
        assignment[animal] = best + 1
        add_aramra_fold_features(stats[best], animal_features[animal])
    return assignment


def add_aramra_fold_features(stats: dict[str, Any], features: dict[str, Any]) -> None:
    stats["animals"] += 1
    stats["cases"] += int(features["cases"])
    stats["pos"] += int(features["pos"])
    stats["time"].update(features["time"])
    stats["pattern"].update(features["pattern"])


def norm_abs(value: float, target: float, floor: float) -> float:
    return abs(float(value) - float(target)) / max(abs(float(target)), floor)


def aramra_fold_score(
    fold_stats: list[dict[str, Any]],
    totals: dict[str, Any],
    features: dict[str, Any],
    fold_idx: int,
    folds: int,
    time_keys: list[str],
    pattern_keys: list[str],
) -> float:
    trial = []
    for idx, stats in enumerate(fold_stats):
        copied = {
            "animals": stats["animals"],
            "cases": stats["cases"],
            "pos": stats["pos"],
            "time": Counter(stats["time"]),
            "pattern": Counter(stats["pattern"]),
        }
        if idx == fold_idx:
            add_aramra_fold_features(copied, features)
        trial.append(copied)
    score = 0.0
    for stats in trial:
        score += 2.0 * norm_abs(stats["animals"], totals["animals"] / folds, 1.0)
        score += 2.0 * norm_abs(stats["cases"], totals["cases"] / folds, 1.0)
        score += 1.0 * norm_abs(stats["pos"], totals["pos"] / folds, 1.0)
        for key in time_keys:
            score += 0.7 * norm_abs(stats["time"][key], totals["time"][key] / folds, 1.0)
        for key in pattern_keys:
            score += 0.3 * norm_abs(stats["pattern"][key], totals["pattern"][key] / folds, 1.0)
    return score


def record_to_row(record: Any) -> dict[str, Any]:
    return {
        "case_id": record.case_id,
        "cohort": record.cohort,
        "animal_id_strict": record.animal_id_strict,
        "animal_id_raw": record.animal_id_raw,
        "animal_family": record.animal_family,
        "time_raw": record.time_raw,
        "timepoint_pattern": record.timepoint_pattern,
        "fold": record.fold or "",
        "revised_status": record.revised_status,
        "image_path": str(record.image_path),
        "round0_label_path": str(record.round0_label_path or ""),
        "round1_label_path": str(record.round1_label_path or ""),
        "eval_label_path": str(record.eval_label_path or ""),
    }


def write_metadata(run_dir: Path, epi_cases: list[Any], ar_cases: list[Any], cfg: dict[str, Any]) -> None:
    source_workspace = resolve_path(cfg["paths"]["source_workspace"])
    if source_workspace is None:
        raise ValueError("paths.source_workspace is required")
    assign_source_folds(epi_cases, source_workspace)
    folds = int(cfg["split"].get("aramra_folds", 5))
    ar_fold_map = make_aramra_folds(ar_cases, folds, int(cfg.get("seed", 20260607)))
    for record in ar_cases:
        record.fold = ar_fold_map[record.animal_id_strict]
    write_csv(run_dir / "metadata" / "metadata_master.csv", [record_to_row(r) for r in [*epi_cases, *ar_cases]])
    write_csv(run_dir / "metadata" / "epibios_cases.csv", [record_to_row(r) for r in epi_cases])
    write_csv(run_dir / "metadata" / "aramra_cases.csv", [record_to_row(r) for r in ar_cases])
    split_rows = []
    for animal, fold in sorted(ar_fold_map.items(), key=lambda item: (item[1], item[0])):
        items = [record for record in ar_cases if record.animal_id_strict == animal]
        split_rows.append(
            {
                "animal_id_strict": animal,
                "aramra_fold": fold,
                "num_cases": len(items),
                "timepoint_pattern": items[0].timepoint_pattern if items else "",
                "timepoints": ";".join(sorted(record.time_raw for record in items)),
                "positive_voxels": sum(aramra_positive_voxels(record) for record in items),
            }
        )
    write_csv(run_dir / "splits" / "aramra_animalwise_folds.csv", split_rows)
    write_csv(
        run_dir / "splits" / "epibios_source_folds.csv",
        [
            {
                "case_id": record.case_id,
                "animal_id_strict": record.animal_id_strict,
                "source_fold": record.fold,
                "time_raw": record.time_raw,
                "revised_status": record.revised_status,
            }
            for record in epi_cases
        ],
    )
    fold_animals = {
        fold: {record.animal_id_strict for record in ar_cases if int(record.fold or 0) == fold}
        for fold in range(1, folds + 1)
    }
    overlaps = []
    for a, b in combinations(range(1, folds + 1), 2):
        overlap = sorted(fold_animals[a] & fold_animals[b])
        if overlap:
            overlaps.append({"fold_a": a, "fold_b": b, "animals": overlap})
    write_json(
        run_dir / "splits" / "split_integrity_report.json",
        {
            "status": "pass" if not overlaps else "fail",
            "aramra_used_for_selection_eval_leakage": False,
            "aramra_fold_overlaps": overlaps,
            "num_epibios_cases": len(epi_cases),
            "num_epibios_animals": len({r.animal_id_strict for r in epi_cases}),
            "num_aramra_cases": len(ar_cases),
            "num_aramra_animals": len({r.animal_id_strict for r in ar_cases}),
            "aramra_fold_summary": split_rows,
        },
    )


def select_device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg["training"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_model(cfg: dict[str, Any]) -> torch.nn.Module:
    return aw.make_model(cfg)


def load_checkpoint_compatible(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        state = payload
        for key in ("model", "model_state", "state_dict"):
            if key in payload:
                state = payload[key]
                break
    else:
        state = payload
    model.load_state_dict(state)


def source_checkpoint_path(cfg: dict[str, Any], fold_id: int, round_index: int = 1) -> Path:
    source_workspace = resolve_path(cfg["paths"]["source_workspace"])
    if source_workspace is None:
        raise ValueError("paths.source_workspace is required")
    return source_workspace / "artifacts" / "checkpoints" / f"round_{round_index}" / f"fold_{fold_id}.pt"


def preflight_checks(cfg: dict[str, Any], run_dir: Path, logger: TeeLogger) -> None:
    update_state(run_dir, current_stage="preflight")
    paths_cfg = cfg.get("paths", {})
    source_workspace = resolve_path(paths_cfg.get("source_workspace"))
    epibios_root = resolve_path(paths_cfg.get("epibios_data_root"))
    aramra_root = resolve_path(paths_cfg.get("aramra_root"))
    source_folds = int(cfg["split"].get("source_folds", 5))
    source_round = int(cfg["selection"].get("source_round", 1))
    checks: list[dict[str, Any]] = []

    def add_check(name: str, path: Optional[Path], required: bool = True) -> None:
        exists = bool(path is not None and path.exists())
        checks.append({"name": name, "path": str(path or ""), "required": required, "exists": exists})

    add_check("source_workspace", source_workspace)
    if source_workspace is not None:
        add_check("source_oof_round1_metrics", source_workspace / "reports" / "round_1" / "oof_case_metrics.csv")
        add_check("source_round0_binary_labels", source_workspace / "artifacts" / "labels" / "binary" / "round_0")
        add_check("source_round1_binary_labels", source_workspace / "artifacts" / "labels" / "binary" / "round_1")
        for fold in range(1, source_folds + 1):
            add_check(f"source_round{source_round}_fold{fold}_checkpoint", source_checkpoint_path(cfg, fold, source_round))
    add_check("epibios_data_root", epibios_root)
    if epibios_root is not None:
        add_check("epibios_imagesTr", epibios_root / "imagesTr")
    add_check("aramra_root", aramra_root)
    if aramra_root is not None:
        images_ts = resolve_path(paths_cfg.get("aramra_images_dir")) or (aramra_root / "imagesTs")
        labels_ts = resolve_path(paths_cfg.get("aramra_labels_dir")) or (aramra_root / "labelsTs")
        add_check("aramra_imagesTs", images_ts, required=False)
        add_check("aramra_labelsTs", labels_ts, required=False)
        if images_ts.exists() and labels_ts.exists():
            image_count = len([path for path in images_ts.glob("*.nii.gz") if not path.name.startswith("._")])
            label_count = len([path for path in labels_ts.glob("*.nii.gz") if not path.name.startswith("._")])
        else:
            image_count = 0
            label_count = 0
        checks.append({"name": "aramra_standard_image_count", "path": str(images_ts), "required": False, "exists": image_count > 0, "count": image_count})
        checks.append({"name": "aramra_standard_label_count", "path": str(labels_ts), "required": False, "exists": label_count > 0, "count": label_count})

    device_cfg = str(cfg["training"].get("device", "auto"))
    if device_cfg == "cuda" and not torch.cuda.is_available():
        checks.append({"name": "cuda_requested_available", "path": "torch.cuda.is_available", "required": True, "exists": False})
    elif device_cfg in {"auto", "cuda"}:
        checks.append({"name": "cuda_available", "path": "torch.cuda.is_available", "required": False, "exists": bool(torch.cuda.is_available())})

    errors = [row for row in checks if bool(row.get("required", True)) and not bool(row.get("exists", False))]
    report = {
        "status": "pass" if not errors else "fail",
        "checks": checks,
        "errors": errors,
    }
    write_json(run_dir / "config" / "preflight_report.json", report)
    if errors:
        for row in errors:
            logger.log(f"preflight missing {row['name']}: {row['path']}")
        raise RuntimeError(f"Preflight failed with {len(errors)} missing required paths; see config/preflight_report.json")
    logger.log("preflight checks passed")
    append_completed_stage(run_dir, "preflight")


def load_source_models(cfg: dict[str, Any], device: torch.device, round_index: int = 1) -> list[torch.nn.Module]:
    folds = int(cfg["split"].get("source_folds", 5))
    models = []
    for fold_id in range(1, folds + 1):
        model = make_model(cfg).to(device)
        load_checkpoint_compatible(model, source_checkpoint_path(cfg, fold_id, round_index), device)
        model.eval()
        models.append(model)
    return models


def top_n_mean(values: np.ndarray, n: int) -> float:
    flat = values.reshape(-1)
    if flat.size == 0:
        return 0.0
    n = max(1, min(int(n), flat.size))
    idx = np.argpartition(flat, flat.size - n)[-n:]
    return float(flat[idx].mean())


def positive_slice_fraction(binary_mask: np.ndarray) -> float:
    positives = np.any(binary_mask.astype(bool), axis=(0, 1))
    return float(positives.mean())


def connected_components_score(binary_mask: np.ndarray, max_components: int = 20) -> tuple[int, float]:
    _, count = ndimage.label(binary_mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
    return int(count), min(int(count), int(max_components)) / float(max_components)


def predict_source_on_aramra(ar_cases: list[Any], cfg: dict[str, Any], run_dir: Path, logger: TeeLogger) -> list[dict[str, Any]]:
    update_state(run_dir, current_stage="source_prediction_scoring")
    device = select_device(cfg)
    models = load_source_models(cfg, device, round_index=int(cfg["selection"].get("source_round", 1)))
    rows: list[dict[str, Any]] = []
    threshold = float(cfg["inference"].get("threshold", 0.5))
    out_root = run_dir / "source_predictions" / "aramra"
    score_root = run_dir / "scores" / "arrays"
    ensure_dir(score_root)
    for idx, record in enumerate(ar_cases, start=1):
        image_vol = aw.load_nifti(record.image_path)
        image = aw.percentile_zscore(image_vol.data)
        ref = aw.load_binary_label(record.eval_label_path)
        fold_probs = []
        for source_fold, model in enumerate(models, start=1):
            prob = aw.predict_volume(model, image, cfg, device)
            fold_probs.append(prob.astype(np.float32))
            if bool(cfg["outputs"].get("save_source_fold_probabilities", False)):
                fold_path = out_root / "fold_probabilities" / f"fold_{source_fold}" / f"{record.case_id}.npz"
                ensure_dir(fold_path.parent)
                np.savez_compressed(fold_path, probability=prob, s=prob)
        stack = np.stack(fold_probs, axis=0)
        prob_mean = stack.mean(axis=0).astype(np.float32)
        prob_var = stack.var(axis=0).astype(np.float32)
        prob_path = out_root / "probabilities" / f"{record.case_id}.npz"
        ensure_dir(prob_path.parent)
        np.savez_compressed(prob_path, probability=prob_mean, s=prob_mean, ensemble_variance=prob_var)
        raw_mask = (prob_mean >= threshold).astype(np.uint8)
        mask_path = out_root / "masks_raw" / f"{record.case_id}.nii.gz"
        if bool(cfg["outputs"].get("save_source_masks", True)):
            aw.save_nifti(mask_path, raw_mask, image_vol.affine, image_vol.header, np.uint8)
        n_vox = max(int(cfg["selection"].get("top_voxel_min", 100)), int(math.ceil(raw_mask.size * float(cfg["selection"].get("top_voxel_fraction", 0.01)))))
        disagreement = top_n_mean(np.abs(ref.astype(np.float32) - prob_mean), n_vox)
        uncertainty = top_n_mean(prob_var, n_vox)
        positive_fraction = float(raw_mask.mean())
        positive_slices = positive_slice_fraction(raw_mask)
        cc_raw, cc_score = connected_components_score(raw_mask, max_components=int(cfg["selection"].get("max_connected_components", 20)))
        cost = (
            1.0
            + float(cfg["selection"].get("positive_voxel_cost_weight", 2.0)) * positive_fraction
            + float(cfg["selection"].get("positive_slice_cost_weight", 1.0)) * positive_slices
            + float(cfg["selection"].get("connected_component_cost_weight", 1.0)) * cc_score
        )
        source_metrics = mask_metrics(raw_mask, ref, spacing_from_header(image_vol.header))
        row = {
            "case_id": record.case_id,
            "animal_id_strict": record.animal_id_strict,
            "time_raw": record.time_raw,
            "timepoint_pattern": record.timepoint_pattern,
            "aramra_fold": record.fold,
            "source_probability_path": str(prob_path),
            "source_mask_path": str(mask_path),
            "disagreement_topmean": disagreement,
            "uncertainty_topmean": uncertainty,
            "review_cost": cost,
            "pred_positive_fraction": positive_fraction,
            "pred_positive_slice_fraction": positive_slices,
            "pred_connected_components": cc_raw,
            "ref_positive_voxels": int(ref.sum()),
            "source_dice": source_metrics["dice"],
            "source_hd95_mm": source_metrics["hd95_mm"],
            "source_assd_mm": source_metrics["assd_mm"],
            "source_abs_rve_percent": source_metrics["abs_rve_percent"],
        }
        rows.append(row)
        if idx % int(cfg["outputs"].get("log_every_cases", 10)) == 0 or idx == len(ar_cases):
            logger.log(f"source predicted/scored ARAMRA {idx}/{len(ar_cases)}")
    write_csv(run_dir / "scores" / "source_scores_case.csv", rows)
    source_metric_rows = []
    for row in rows:
        record = next(r for r in ar_cases if r.case_id == row["case_id"])
        mask_path = Path(row["source_mask_path"])
        if mask_path.exists():
            mask = aw.load_binary_label(mask_path)
        else:
            prob = aw.load_probability(Path(row["source_probability_path"]))
            mask = (prob >= threshold).astype(np.uint8)
        ref = aw.load_binary_label(record.eval_label_path)
        source_metric_rows.append(metric_row(record, mask, ref, "raw", "source_only", "ensemble"))
    write_csv(run_dir / "metrics" / "source_only_aramra_case_metrics.csv", source_metric_rows)
    write_summary_tables(run_dir, source_metric_rows, "source_only_aramra")
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    append_completed_stage(run_dir, "source_prediction_scoring")
    return rows


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1.0e-12:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def animal_score_table(score_rows: list[dict[str, Any]], candidate_animals: set[str], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_rows = [row for row in score_rows if str(row["animal_id_strict"]) in candidate_animals]
    d_norm = minmax([float(row["disagreement_topmean"]) for row in candidate_rows])
    u_norm = minmax([float(row["uncertainty_topmean"]) for row in candidate_rows])
    c_norm = minmax([float(row["review_cost"]) for row in candidate_rows])
    case_rows = []
    for row, d_bar, u_bar, c_bar in zip(candidate_rows, d_norm, u_norm, c_norm):
        utility = (
            float(cfg["selection"].get("disagreement_weight", 0.70)) * d_bar
            + float(cfg["selection"].get("uncertainty_weight", 0.30)) * u_bar
        ) / (1.0 + c_bar)
        out = dict(row)
        out.update({"d_bar": d_bar, "u_bar": u_bar, "c_bar": c_bar, "utility": utility})
        case_rows.append(out)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row["animal_id_strict"])].append(row)
    animal_rows = []
    for animal, items in grouped.items():
        utilities = [float(row["utility"]) for row in items]
        disagreements = [float(row["disagreement_topmean"]) for row in items]
        uncertainties = [float(row["uncertainty_topmean"]) for row in items]
        source_dice = [float(row["source_dice"]) for row in items]
        ref_volumes = [int(row["ref_positive_voxels"]) for row in items]
        animal_rows.append(
            {
                "animal_id_strict": animal,
                "animal_utility_max": float(max(utilities)),
                "animal_utility_mean": float(np.mean(utilities)),
                "animal_disagreement_max": float(max(disagreements)),
                "animal_uncertainty_max": float(max(uncertainties)),
                "animal_source_dice_mean": float(np.mean(source_dice)),
                "num_cases": len(items),
                "case_ids": ";".join(str(row["case_id"]) for row in items),
                "timepoints": ";".join(sorted(str(row["time_raw"]) for row in items)),
                "has_m5": int(any(str(row["time_raw"]) == "M5" for row in items)),
                "min_ref_positive_voxels": int(min(ref_volumes)) if ref_volumes else 0,
                "total_ref_positive_voxels": int(sum(ref_volumes)),
            }
        )
    return sorted(animal_rows, key=lambda row: (-float(row["animal_utility_max"]), str(row["animal_id_strict"])))


def select_animals_random_stratified(animals: list[dict[str, Any]], budget: int, seed: int) -> list[dict[str, Any]]:
    if budget >= len(animals):
        return [{**row, "selection_reason": "all_available"} for row in animals]
    rng = np.random.default_rng(seed)
    shuffled = list(animals)
    rng.shuffle(shuffled)
    target_m5 = sum(int(row["has_m5"]) for row in animals) / max(len(animals), 1)
    target_cases = sum(int(row["num_cases"]) for row in animals) / max(len(animals), 1)
    target_log_volume = float(np.mean([math.log1p(float(row["total_ref_positive_voxels"])) for row in animals])) if animals else 0.0
    selected: list[dict[str, Any]] = []
    remaining = list(shuffled)
    while len(selected) < budget and remaining:
        best_idx = 0
        best_score = float("inf")
        for idx, row in enumerate(remaining):
            trial = selected + [row]
            m5_rate = sum(int(item["has_m5"]) for item in trial) / len(trial)
            case_mean = sum(int(item["num_cases"]) for item in trial) / len(trial)
            log_volume_mean = float(np.mean([math.log1p(float(item["total_ref_positive_voxels"])) for item in trial]))
            score = (
                abs(m5_rate - target_m5)
                + 0.1 * abs(case_mean - target_cases)
                + 0.1 * abs(log_volume_mean - target_log_volume)
                + float(rng.random()) * 1.0e-6
            )
            if score < best_score:
                best_score = score
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return [{**row, "selection_reason": "random_stratified"} for row in selected]


def select_animals_ranked(animals: list[dict[str, Any]], budget: int, key: str, reason: str) -> list[dict[str, Any]]:
    ordered = sorted(animals, key=lambda row: (-float(row[key]), str(row["animal_id_strict"])))
    return [{**row, "selection_reason": reason} for row in ordered[: min(budget, len(ordered))]]


def select_animals_balanced_utility(animals: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    if budget >= len(animals):
        return [{**row, "selection_reason": "all_available"} for row in animals]
    ordered = sorted(animals, key=lambda row: float(row["animal_utility_max"]))
    utilities = [float(row["animal_utility_max"]) for row in ordered]
    q10 = float(np.quantile(utilities, 0.10))
    q50 = float(np.quantile(utilities, 0.50))
    q95 = float(np.quantile(utilities, 0.95))
    n_main = max(1, int(round(0.50 * budget)))
    n_rep = max(1, int(round(0.25 * budget))) if budget >= 4 else 0
    n_risk = max(0, budget - n_main - n_rep)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_from(pool: list[dict[str, Any]], count: int, reason: str) -> None:
        for row in pool:
            if len(selected) >= budget or count <= 0:
                break
            animal = str(row["animal_id_strict"])
            if animal in selected_ids:
                continue
            selected_ids.add(animal)
            selected.append({**row, "selection_reason": reason})
            count -= 1

    main_pool = [
        row for row in sorted(animals, key=lambda row: (-float(row["animal_utility_max"]), str(row["animal_id_strict"])))
        if q50 <= float(row["animal_utility_max"]) <= q95
    ]
    add_from(main_pool, n_main, "balanced_mid_high_non_extreme")
    rep_pool = [
        row for row in sorted(animals, key=lambda row: (float(row["animal_utility_max"]), str(row["animal_id_strict"])))
        if q10 <= float(row["animal_utility_max"]) <= q50
    ]
    add_from(rep_pool, n_rep, "balanced_representative_low_mid")
    small_cutoff = float(np.quantile([float(row["min_ref_positive_voxels"]) for row in animals], 0.25))
    risk_pool = [
        row for row in sorted(animals, key=lambda row: (-int(row["has_m5"]), float(row["min_ref_positive_voxels"]), -float(row["animal_utility_max"])))
        if int(row["has_m5"]) or float(row["min_ref_positive_voxels"]) <= small_cutoff
    ]
    add_from(risk_pool, n_risk, "balanced_m5_small_lesion_risk")
    fill_pool = sorted(animals, key=lambda row: (-float(row["animal_utility_max"]), str(row["animal_id_strict"])))
    add_from(fill_pool, budget - len(selected), "balanced_fill_high_utility")
    return selected


def planned_experiments(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    budgets = cfg["selection"].get("budgets_animals", [10])
    methods = cfg["selection"].get("methods", ["random", "disagreement", "balanced_utility"])
    out = [{"method": "source_only", "budget": 0, "budget_label": "0"}]
    for budget in budgets:
        for method in methods:
            out.append({"method": str(method), "budget": budget, "budget_label": str(budget)})
    if bool(cfg["selection"].get("run_all_train_pool", True)):
        out.append({"method": "all_train_pool", "budget": "all", "budget_label": "all"})
    return out


def select_for_fold(
    outer_fold: int,
    ar_cases: list[Any],
    score_rows: list[dict[str, Any]],
    experiment: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_animals = {record.animal_id_strict for record in ar_cases if int(record.fold) != outer_fold}
    animals = animal_score_table(score_rows, candidate_animals, cfg)
    method = str(experiment["method"])
    if method == "source_only":
        return [], animals, []
    budget = len(animals) if experiment["budget"] == "all" else min(int(experiment["budget"]), len(animals))
    seed = int(cfg.get("seed", 20260607)) + outer_fold * 1009 + budget * 17
    if method == "random":
        selected = select_animals_random_stratified(animals, budget, seed)
    elif method == "disagreement":
        selected = select_animals_ranked(animals, budget, "animal_disagreement_max", "high_disagreement")
    elif method == "uncertainty":
        selected = select_animals_ranked(animals, budget, "animal_uncertainty_max", "high_uncertainty")
    elif method == "balanced_utility":
        selected = select_animals_balanced_utility(animals, budget)
    elif method == "all_train_pool":
        selected = [{**row, "selection_reason": "all_train_pool"} for row in animals]
    else:
        raise ValueError(f"Unsupported selection method: {method}")
    selected_ids = [str(row["animal_id_strict"]) for row in selected]
    return selected_ids, animals, selected


def write_selection_tables(run_dir: Path, ar_cases: list[Any], score_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[tuple[int, str, str], list[str]]:
    update_state(run_dir, current_stage="selection")
    folds = int(cfg["split"].get("aramra_folds", 5))
    selection_map: dict[tuple[int, str, str], list[str]] = {}
    all_animal_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for outer_fold in range(1, folds + 1):
        for experiment in planned_experiments(cfg):
            selected_ids, animals, selected = select_for_fold(outer_fold, ar_cases, score_rows, experiment, cfg)
            method = str(experiment["method"])
            budget_label = str(experiment["budget_label"])
            heldout_animals = {record.animal_id_strict for record in ar_cases if int(record.fold) == outer_fold}
            leakage = sorted(set(selected_ids) & heldout_animals)
            if leakage:
                raise RuntimeError(
                    f"ARAMRA selection leakage for outer_fold={outer_fold} "
                    f"method={method} budget={budget_label}: {leakage}"
                )
            selection_map[(outer_fold, method, budget_label)] = selected_ids
            for row in animals:
                all_animal_rows.append({"outer_fold": outer_fold, "method": method, "budget": budget_label, **row})
            for rank, row in enumerate(selected, start=1):
                selected_rows.append({"outer_fold": outer_fold, "method": method, "budget": budget_label, "rank": rank, **row})
    write_csv(run_dir / "selection" / "candidate_animal_scores_by_fold.csv", all_animal_rows)
    write_csv(run_dir / "selection" / "selected_animals.csv", selected_rows)
    append_completed_stage(run_dir, "selection")
    return selection_map


def prepare_training_case(record: Any, label_kind: str, cfg: dict[str, Any], case_weight: float) -> Any:
    image = aw.percentile_zscore(aw.load_nifti(record.image_path).data)
    if label_kind == "epibios_round1":
        label_path = record.round1_label_path
    elif label_kind == "aramra_eval":
        label_path = record.eval_label_path
    else:
        raise ValueError(label_kind)
    if label_path is None:
        raise ValueError(f"Missing label path for case={record.case_id} label_kind={label_kind}")
    binary = aw.load_binary_label(label_path)
    target = binary.astype(np.float32)
    return aw.PreparedCase(
        record=record,
        image=image,
        target=target,
        eval_label=binary.astype(np.uint8),
        voxel_weight=np.ones_like(target, dtype=np.float32),
        case_weight=float(case_weight),
        positive_mask=binary.astype(np.uint8),
    )


def train_adapted_source_models(
    outer_fold: int,
    method: str,
    budget_label: str,
    selected_animals: list[str],
    epi_cases: list[Any],
    ar_cases: list[Any],
    cfg: dict[str, Any],
    run_dir: Path,
    logger: TeeLogger,
) -> list[Path]:
    device = select_device(cfg)
    source_folds = int(cfg["split"].get("source_folds", 5))
    ckpts: list[Path] = []
    if method == "source_only":
        return [source_checkpoint_path(cfg, source_fold, int(cfg["selection"].get("source_round", 1))) for source_fold in range(1, source_folds + 1)]
    exp_key = experiment_key(method, budget_label)
    fold_dir = ensure_dir(run_dir / "checkpoints" / exp_key / f"outer_fold_{outer_fold}")
    selected_set = set(selected_animals)
    selected_ar = [record for record in ar_cases if record.animal_id_strict in selected_set]
    phase_cfg = dict(cfg["training"].get("adaptation", {}))
    if "epochs" not in phase_cfg:
        phase_cfg["epochs"] = int(cfg["training"].get("epochs", 20))
    if "lr" not in phase_cfg:
        phase_cfg["lr"] = float(cfg["training"].get("lr", 2.0e-5))
    if "weight_decay" not in phase_cfg:
        phase_cfg["weight_decay"] = float(cfg["training"].get("weight_decay", 1.0e-4))
    for source_fold in range(1, source_folds + 1):
        update_state(run_dir, current_stage=f"adapt_{exp_key}_outer{outer_fold}_source{source_fold}")
        fold_log = TeeLogger(run_dir / "logs" / "adaptation" / exp_key / f"outer_{outer_fold}_source_{source_fold}.log")
        try:
            model = make_model(cfg).to(device)
            load_checkpoint_compatible(model, source_checkpoint_path(cfg, source_fold, int(cfg["selection"].get("source_round", 1))), device)
            if str(cfg["training"].get("epibios_replay_mode", "source_fold_train_only")) == "source_fold_train_only":
                epi_replay = [record for record in epi_cases if int(record.fold) != source_fold]
            else:
                epi_replay = list(epi_cases)
            train_prepared = [
                prepare_training_case(record, "epibios_round1", cfg, float(cfg["training"].get("epibios_case_weight", 1.0)))
                for record in epi_replay
            ]
            train_prepared.extend(
                prepare_training_case(record, "aramra_eval", cfg, float(cfg["training"].get("aramra_case_weight", 1.0)))
                for record in selected_ar
            )
            fold_log.log(
                f"start adaptation method={method} budget={budget_label} outer_fold={outer_fold} "
                f"source_fold={source_fold} epi_replay={len(epi_replay)} selected_aramra_cases={len(selected_ar)}"
            )
            best_tmp, last_tmp, rows = aw.fit_model(
                model,
                train_prepared,
                [],
                cfg,
                phase_cfg,
                device,
                fold_log,
                f"{exp_key}_outer{outer_fold}",
                source_fold,
            )
            best_path = fold_dir / f"source_fold_{source_fold}_best.pt"
            last_path = fold_dir / f"source_fold_{source_fold}_last.pt"
            shutil.copy2(best_tmp, best_path)
            shutil.copy2(last_tmp, last_path)
            write_csv(run_dir / "metrics" / "training" / exp_key / f"outer_{outer_fold}_source_{source_fold}_train.csv", rows)
            ckpts.append(best_path)
            fold_log.log(f"complete adaptation checkpoint={best_path}")
        finally:
            fold_log.close()
    return ckpts


def experiment_key(method: str, budget_label: str) -> str:
    safe_budget = str(budget_label).replace("/", "_")
    return f"{method}_budget_{safe_budget}"


def predict_ensemble(checkpoints: list[Path], record: Any, cfg: dict[str, Any], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    image = aw.percentile_zscore(aw.load_nifti(record.image_path).data)
    probs = []
    for checkpoint in checkpoints:
        model = make_model(cfg).to(device)
        load_checkpoint_compatible(model, checkpoint, device)
        model.eval()
        probs.append(aw.predict_volume(model, image, cfg, device))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    stack = np.stack(probs, axis=0).astype(np.float32)
    return stack.mean(axis=0).astype(np.float32), stack.var(axis=0).astype(np.float32)


def predict_ensemble_for_records(
    checkpoints: list[Path],
    records: list[Any],
    cfg: dict[str, Any],
    device: torch.device,
    logger: TeeLogger | None = None,
    progress_label: str = "",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not records:
        return {}
    sums: dict[str, np.ndarray] = {}
    sq_sums: dict[str, np.ndarray] = {}
    count = 0
    for ckpt_idx, checkpoint in enumerate(checkpoints, start=1):
        model = make_model(cfg).to(device)
        load_checkpoint_compatible(model, checkpoint, device)
        model.eval()
        for rec_idx, record in enumerate(records, start=1):
            image = aw.percentile_zscore(aw.load_nifti(record.image_path).data)
            prob = aw.predict_volume(model, image, cfg, device).astype(np.float32)
            if record.case_id not in sums:
                sums[record.case_id] = np.zeros_like(prob, dtype=np.float32)
                sq_sums[record.case_id] = np.zeros_like(prob, dtype=np.float32)
            sums[record.case_id] += prob
            sq_sums[record.case_id] += prob * prob
            if logger is not None and rec_idx == len(records):
                logger.log(f"predicted {progress_label} checkpoint={ckpt_idx}/{len(checkpoints)} cases={rec_idx}/{len(records)}")
        count += 1
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    denom = float(max(count, 1))
    for case_id in sums:
        mean = (sums[case_id] / denom).astype(np.float32)
        var = np.maximum((sq_sums[case_id] / denom) - (mean * mean), 0.0).astype(np.float32)
        out[case_id] = (mean, var)
    return out


def evaluate_experiment_fold(
    outer_fold: int,
    method: str,
    budget_label: str,
    checkpoints: list[Path],
    ar_cases: list[Any],
    epi_cases: list[Any],
    cfg: dict[str, Any],
    run_dir: Path,
    logger: TeeLogger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = select_device(cfg)
    exp_key = experiment_key(method, budget_label)
    threshold = float(cfg["inference"].get("threshold", 0.5))
    post_values = [int(v) for v in cfg["inference"].get("postprocess_min_component_voxels", [2])]
    ar_rows: list[dict[str, Any]] = []
    epi_rows: list[dict[str, Any]] = []
    heldout_ar = [record for record in ar_cases if int(record.fold) == outer_fold]
    pred_root = run_dir / "predictions" / "aramra_oof" / exp_key / f"outer_fold_{outer_fold}"
    fold_predictions = predict_ensemble_for_records(
        checkpoints,
        heldout_ar,
        cfg,
        device,
        logger,
        f"ARAMRA {exp_key} outer_fold={outer_fold}",
    )
    for idx, record in enumerate(heldout_ar, start=1):
        image_vol = aw.load_nifti(record.image_path)
        prob, var = fold_predictions[record.case_id]
        prob_path = pred_root / "probabilities" / f"{record.case_id}.npz"
        ensure_dir(prob_path.parent)
        np.savez_compressed(prob_path, probability=prob, s=prob, ensemble_variance=var)
        ref = aw.load_binary_label(record.eval_label_path)
        variants = {"raw": (prob >= threshold).astype(np.uint8)}
        for min_vox in post_values:
            variants[f"post_min{min_vox}"] = aw.postprocess_probability_map(prob, threshold, min_vox, False)
        for variant, mask in variants.items():
            mask_path = pred_root / f"masks_{variant}" / f"{record.case_id}.nii.gz"
            if bool(cfg["outputs"].get("save_masks", True)):
                aw.save_nifti(mask_path, mask, image_vol.affine, image_vol.header, np.uint8)
            row = metric_row(record, mask, ref, variant, exp_key, outer_fold)
            row.update({"method": method, "budget": budget_label, "outer_fold": outer_fold, "prediction_path": str(mask_path), "probability_path": str(prob_path)})
            ar_rows.append(row)
        if idx % int(cfg["outputs"].get("log_every_cases", 10)) == 0 or idx == len(heldout_ar):
            logger.log(f"evaluated ARAMRA {exp_key} outer_fold={outer_fold} {idx}/{len(heldout_ar)}")
    run_retention = bool(cfg["stages"].get("eval_epibios_retention", True))
    if method == "source_only" and outer_fold != 1:
        run_retention = False
    if run_retention:
        source_folds = int(cfg["split"].get("source_folds", 5))
        for source_fold in range(1, source_folds + 1):
            model = make_model(cfg).to(device)
            load_checkpoint_compatible(model, checkpoints[source_fold - 1], device)
            model.eval()
            holdout_epi = [record for record in epi_cases if int(record.fold) == source_fold]
            epi_root = run_dir / "predictions" / "epibios_retention" / exp_key / f"outer_fold_{outer_fold}" / f"source_fold_{source_fold}"
            for record in holdout_epi:
                image_vol = aw.load_nifti(record.image_path)
                image = aw.percentile_zscore(image_vol.data)
                prob = aw.predict_volume(model, image, cfg, device)
                ref = aw.load_binary_label(record.round1_label_path)
                mask = (prob >= threshold).astype(np.uint8)
                mask_path = epi_root / "masks_raw" / f"{record.case_id}.nii.gz"
                if bool(cfg["outputs"].get("save_retention_masks", False)):
                    aw.save_nifti(mask_path, mask, image_vol.affine, image_vol.header, np.uint8)
                row = metric_row(record, mask, ref, "raw", exp_key, source_fold)
                row.update({"method": method, "budget": budget_label, "outer_fold": outer_fold, "source_fold": source_fold, "prediction_path": str(mask_path)})
                epi_rows.append(row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return ar_rows, epi_rows


def run_main_experiments(
    epi_cases: list[Any],
    ar_cases: list[Any],
    score_rows: list[dict[str, Any]],
    selection_map: dict[tuple[int, str, str], list[str]],
    cfg: dict[str, Any],
    run_dir: Path,
    logger: TeeLogger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_ar_rows: list[dict[str, Any]] = []
    all_epi_rows: list[dict[str, Any]] = []
    folds = int(cfg["split"].get("aramra_folds", 5))
    for experiment in planned_experiments(cfg):
        method = str(experiment["method"])
        budget_label = str(experiment["budget_label"])
        exp_key = experiment_key(method, budget_label)
        for outer_fold in range(1, folds + 1):
            update_state(run_dir, current_stage=f"main_{exp_key}_outer{outer_fold}")
            selected = selection_map.get((outer_fold, method, budget_label), [])
            checkpoints = train_adapted_source_models(outer_fold, method, budget_label, selected, epi_cases, ar_cases, cfg, run_dir, logger)
            ar_rows, epi_rows = evaluate_experiment_fold(outer_fold, method, budget_label, checkpoints, ar_cases, epi_cases, cfg, run_dir, logger)
            all_ar_rows.extend(ar_rows)
            all_epi_rows.extend(epi_rows)
            write_csv(run_dir / "metrics" / "aramra_oof_case_metrics.partial.csv", all_ar_rows)
            write_csv(run_dir / "metrics" / "epibios_retention_case_metrics.partial.csv", all_epi_rows)
    write_csv(run_dir / "metrics" / "aramra_oof_case_metrics.csv", all_ar_rows)
    write_csv(run_dir / "metrics" / "epibios_retention_case_metrics.csv", all_epi_rows)
    write_summary_tables(run_dir, all_ar_rows, "aramra_oof")
    write_summary_tables(run_dir, all_epi_rows, "epibios_retention")
    append_completed_stage(run_dir, "main_experiments")
    return all_ar_rows, all_epi_rows


def run_aux_aramra_internal_oof(ar_cases: list[Any], cfg: dict[str, Any], run_dir: Path, logger: TeeLogger) -> list[dict[str, Any]]:
    if not bool(cfg["stages"].get("run_aux_aramra_internal_oof", True)):
        return []
    update_state(run_dir, current_stage="aux_aramra_internal_oof")
    device = select_device(cfg)
    folds = int(cfg["split"].get("aramra_folds", 5))
    phase_cfg = dict(cfg["training"].get("aux_aramra", cfg["training"].get("adaptation", {})))
    phase_cfg.setdefault("epochs", int(cfg["training"].get("aux_epochs", 50)))
    phase_cfg.setdefault("lr", float(cfg["training"].get("aux_lr", 1.0e-4)))
    phase_cfg.setdefault("weight_decay", float(cfg["training"].get("weight_decay", 1.0e-4)))
    rows: list[dict[str, Any]] = []
    threshold = float(cfg["inference"].get("threshold", 0.5))
    for outer_fold in range(1, folds + 1):
        fold_log = TeeLogger(run_dir / "logs" / "aux_aramra_internal" / f"outer_fold_{outer_fold}.log")
        try:
            model = make_model(cfg).to(device)
            train_records = [record for record in ar_cases if int(record.fold) != outer_fold]
            holdout_records = [record for record in ar_cases if int(record.fold) == outer_fold]
            train_prepared = [
                prepare_training_case(record, "aramra_eval", cfg, float(cfg["training"].get("aramra_case_weight", 1.0)))
                for record in train_records
            ]
            fold_log.log(f"start aux ARAMRA internal fold={outer_fold} train_cases={len(train_records)} holdout_cases={len(holdout_records)}")
            best_tmp, last_tmp, train_rows = aw.fit_model(
                model,
                train_prepared,
                [],
                cfg,
                phase_cfg,
                device,
                fold_log,
                "aux_aramra_internal",
                outer_fold,
            )
            ckpt_dir = ensure_dir(run_dir / "checkpoints" / "aux_aramra_internal")
            best_path = ckpt_dir / f"outer_fold_{outer_fold}_best.pt"
            shutil.copy2(best_tmp, best_path)
            shutil.copy2(last_tmp, ckpt_dir / f"outer_fold_{outer_fold}_last.pt")
            write_csv(run_dir / "metrics" / "training" / "aux_aramra_internal" / f"outer_fold_{outer_fold}_train.csv", train_rows)
            load_checkpoint_compatible(model, best_path, device)
            pred_root = run_dir / "predictions" / "aux_aramra_internal_oof" / f"outer_fold_{outer_fold}"
            for record in holdout_records:
                image_vol = aw.load_nifti(record.image_path)
                image = aw.percentile_zscore(image_vol.data)
                prob = aw.predict_volume(model, image, cfg, device)
                ref = aw.load_binary_label(record.eval_label_path)
                mask = (prob >= threshold).astype(np.uint8)
                prob_path = pred_root / "probabilities" / f"{record.case_id}.npz"
                ensure_dir(prob_path.parent)
                np.savez_compressed(prob_path, probability=prob, s=prob)
                mask_path = pred_root / "masks_raw" / f"{record.case_id}.nii.gz"
                if bool(cfg["outputs"].get("save_masks", True)):
                    aw.save_nifti(mask_path, mask, image_vol.affine, image_vol.header, np.uint8)
                row = metric_row(record, mask, ref, "raw", "aux_aramra_internal", outer_fold)
                row.update({"method": "aux_aramra_internal", "budget": "all_train_folds", "outer_fold": outer_fold, "prediction_path": str(mask_path), "probability_path": str(prob_path)})
                rows.append(row)
            fold_log.log(f"complete aux ARAMRA internal fold={outer_fold}")
        finally:
            fold_log.close()
    write_csv(run_dir / "metrics" / "aux_aramra_internal_oof_case_metrics.csv", rows)
    write_summary_tables(run_dir, rows, "aux_aramra_internal_oof")
    write_aux_difficulty_crosswalk(run_dir)
    append_completed_stage(run_dir, "aux_aramra_internal_oof")
    return rows


def spacing_from_header(header: nib.Nifti1Header) -> tuple[float, float, float]:
    zooms = header.get_zooms()[:3]
    return (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def binary_surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask.astype(bool), ~eroded)


def surface_distances(pred: np.ndarray, ref: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    pred_surface = binary_surface(pred)
    ref_surface = binary_surface(ref)
    if not pred_surface.any() or not ref_surface.any():
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    dt_to_ref = ndimage.distance_transform_edt(~ref_surface, sampling=spacing)
    dt_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    return dt_to_ref[pred_surface], dt_to_pred[ref_surface]


def safe_div(num: float, den: float, empty_value: float = float("nan")) -> float:
    if den == 0:
        return empty_value
    return float(num / den)


def mask_metrics(pred: np.ndarray, ref: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, Any]:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    tn = int(np.logical_and(~pred, ~ref).sum())
    pred_voxels = tp + fp
    ref_voxels = tp + fn
    total = tp + fp + fn + tn
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    pred_volume = pred_voxels * voxel_volume
    ref_volume = ref_voxels * voxel_volume
    dice = 1.0 if pred_voxels + ref_voxels == 0 else 2.0 * tp / (pred_voxels + ref_voxels)
    jaccard = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    precision = safe_div(tp, pred_voxels, 1.0 if ref_voxels == 0 else 0.0)
    sensitivity = safe_div(tp, ref_voxels, 1.0 if pred_voxels == 0 else 0.0)
    specificity = safe_div(tn, tn + fp, 1.0)
    volume_similarity = 1.0 if pred_voxels + ref_voxels == 0 else 1.0 - abs(pred_voxels - ref_voxels) / (pred_voxels + ref_voxels)
    rve = safe_div(pred_volume - ref_volume, ref_volume, 0.0 if pred_volume == 0 else float("nan"))
    abs_rve = abs(rve) if not math.isnan(rve) else float("nan")
    agreement = safe_div(tp + tn, total, float("nan"))
    pred_rate = safe_div(pred_voxels, total, 0.0)
    ref_rate = safe_div(ref_voxels, total, 0.0)
    expected = pred_rate * ref_rate + (1.0 - pred_rate) * (1.0 - ref_rate)
    kappa = safe_div(agreement - expected, 1.0 - expected, float("nan"))
    if not pred.any() and not ref.any():
        hd95 = 0.0
        assd = 0.0
        surface_dice_1mm = 1.0
    elif not pred.any() or not ref.any():
        hd95 = float("nan")
        assd = float("nan")
        surface_dice_1mm = 0.0
    else:
        d_pred_ref, d_ref_pred = surface_distances(pred, ref, spacing)
        distances = np.concatenate([d_pred_ref, d_ref_pred])
        hd95 = float(np.percentile(distances, 95)) if distances.size else float("nan")
        assd = float(np.mean(distances)) if distances.size else float("nan")
        surface_dice_1mm = safe_div(int((d_pred_ref <= 1.0).sum() + (d_ref_pred <= 1.0).sum()), int(d_pred_ref.size + d_ref_pred.size), float("nan"))
    return {
        "dice": float(dice),
        "jaccard": float(jaccard),
        "hd95_mm": hd95,
        "assd_mm": assd,
        "rve_percent": rve * 100.0 if not math.isnan(rve) else float("nan"),
        "abs_rve_percent": abs_rve * 100.0 if not math.isnan(abs_rve) else float("nan"),
        "volume_similarity": float(volume_similarity),
        "surface_dice_1mm": float(surface_dice_1mm),
        "voxel_agreement": float(agreement),
        "cohen_kappa": float(kappa),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "tn_voxels": tn,
        "pred_positive_voxels": pred_voxels,
        "ref_positive_voxels": ref_voxels,
        "pred_volume_mm3": pred_volume,
        "ref_volume_mm3": ref_volume,
        "signed_volume_error_mm3": pred_volume - ref_volume,
        "absolute_volume_error_mm3": abs(pred_volume - ref_volume),
        "voxel_volume_mm3": voxel_volume,
    }


def component_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    pred_labeled, pred_n = ndimage.label(pred.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
    ref_labeled, ref_n = ndimage.label(ref.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
    pred_hit = 0
    for idx in range(1, int(pred_n) + 1):
        if np.logical_and(pred_labeled == idx, ref).any():
            pred_hit += 1
    ref_hit = 0
    for idx in range(1, int(ref_n) + 1):
        if np.logical_and(ref_labeled == idx, pred).any():
            ref_hit += 1
    lesion_precision = pred_hit / pred_n if pred_n else (1.0 if ref_n == 0 else 0.0)
    lesion_recall = ref_hit / ref_n if ref_n else (1.0 if pred_n == 0 else 0.0)
    lesion_f1 = 2.0 * lesion_precision * lesion_recall / (lesion_precision + lesion_recall) if lesion_precision + lesion_recall > 0 else 0.0
    return {
        "pred_components": int(pred_n),
        "ref_components": int(ref_n),
        "lesion_precision": float(lesion_precision),
        "lesion_recall": float(lesion_recall),
        "lesion_f1": float(lesion_f1),
    }


def metric_row(record: Any, pred: np.ndarray, ref: np.ndarray, variant: str, experiment: str, fold: Any) -> dict[str, Any]:
    spacing = spacing_from_header(aw.load_nifti(record.image_path).header)
    metrics = mask_metrics(pred, ref, spacing)
    metrics.update(component_metrics(pred, ref))
    row = {
        "experiment": experiment,
        "fold": fold,
        "postprocess_variant": variant,
        "case_id": record.case_id,
        "cohort": record.cohort,
        "animal_id_strict": record.animal_id_strict,
        "animal_id_raw": record.animal_id_raw,
        "animal_family": record.animal_family,
        "time_raw": record.time_raw,
        "timepoint_pattern": record.timepoint_pattern,
    }
    row.update(metrics)
    return row


def dice_from_counts(tp: int, pred_sum: int, ref_sum: int) -> float:
    denom = pred_sum + ref_sum
    if denom == 0:
        return 1.0
    return float(2.0 * tp / denom)


def group_by(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    by_animal = group_by(rows, ["animal_id_strict"])
    tp = sum(int(row["tp_voxels"]) for row in rows)
    pred = sum(int(row["pred_positive_voxels"]) for row in rows)
    ref = sum(int(row["ref_positive_voxels"]) for row in rows)
    metric_names = [
        "dice",
        "jaccard",
        "hd95_mm",
        "assd_mm",
        "rve_percent",
        "abs_rve_percent",
        "volume_similarity",
        "surface_dice_1mm",
        "cohen_kappa",
        "precision",
        "sensitivity",
        "lesion_f1",
        "absolute_volume_error_mm3",
    ]
    out: dict[str, Any] = {
        "num_cases": len(rows),
        "num_animals": len(by_animal),
        "pooled_dice": dice_from_counts(tp, pred, ref),
        "animal_macro_dice": float(np.mean([np.mean([float(row["dice"]) for row in animal_rows]) for animal_rows in by_animal.values()])),
    }
    for metric in metric_names:
        values = [float(row[metric]) for row in rows if row.get(metric, "") != "" and not math.isnan(float(row[metric]))]
        out[f"mean_{metric}"] = float(np.mean(values)) if values else float("nan")
        out[f"median_{metric}"] = float(np.median(values)) if values else float("nan")
    return out


def write_summary_tables(run_dir: Path, rows: list[dict[str, Any]], prefix: str) -> None:
    summary_rows: list[dict[str, Any]] = []
    group_keys = ["experiment", "method", "budget", "postprocess_variant"]
    existing_group_keys = [key for key in group_keys if rows and key in rows[0]]
    for key, items in sorted(group_by(rows, existing_group_keys).items()):
        base = {name: value for name, value in zip(existing_group_keys, key)}
        summary_rows.append({**base, "group": "overall", **aggregate_rows(items)})
    for key, items in sorted(group_by(rows, existing_group_keys + ["time_raw"]).items()):
        base = {name: value for name, value in zip(existing_group_keys + ["time_raw"], key)}
        summary_rows.append({**base, "group": f"time={base.pop('time_raw')}", **aggregate_rows(items)})
    write_csv(run_dir / "metrics" / f"{prefix}_summary.csv", summary_rows)
    write_json(run_dir / "metrics" / f"{prefix}_summary.json", {"summary": summary_rows})


def write_aux_difficulty_crosswalk(run_dir: Path) -> None:
    source_path = run_dir / "metrics" / "source_only_aramra_case_metrics.csv"
    aux_path = run_dir / "metrics" / "aux_aramra_internal_oof_case_metrics.csv"
    if not source_path.exists() or not aux_path.exists():
        return
    source = {row["case_id"]: row for row in read_csv(source_path)}
    rows = []
    for aux in read_csv(aux_path):
        src = source.get(aux["case_id"])
        if not src:
            continue
        source_dice = float(src["dice"])
        aux_dice = float(aux["dice"])
        if source_dice < 0.55 and aux_dice >= 0.65:
            category = "transfer_hard_internal_easier"
        elif source_dice < 0.55 and aux_dice < 0.65:
            category = "hard_both_source_and_internal"
        elif source_dice >= 0.55 and aux_dice < 0.65:
            category = "internal_hard_not_source_specific"
        else:
            category = "easier_both"
        rows.append(
            {
                "case_id": aux["case_id"],
                "animal_id_strict": aux["animal_id_strict"],
                "time_raw": aux["time_raw"],
                "source_only_dice": source_dice,
                "aux_internal_dice": aux_dice,
                "aux_minus_source_dice": aux_dice - source_dice,
                "difficulty_category": category,
            }
        )
    write_csv(run_dir / "metrics" / "aux_source_internal_difficulty_crosswalk.csv", rows)


def write_report(run_dir: Path, cfg: dict[str, Any]) -> None:
    lines = [
        "# Next ARAMRA Model-guided Selection Fine-tuning Report",
        "",
        "## Scope",
        "",
        "This run implements the plan in `project plan/1.txt`: source EpiBios-refined prediction on ARAMRA, model-guided ARAMRA animal selection, selected-label fine-tuning with EpiBios replay, ARAMRA animal-level OOF evaluation, EpiBios retention evaluation, and auxiliary ARAMRA-internal OOF analysis when enabled.",
        "",
    ]
    integrity_path = run_dir / "splits" / "split_integrity_report.json"
    preflight_path = run_dir / "config" / "preflight_report.json"
    if preflight_path.exists():
        preflight = read_json(preflight_path)
        checks = preflight.get("checks", [])
        lines += [
            "## Preflight",
            "",
            f"- Status: `{preflight.get('status')}`",
            f"- Checks: {len(checks)}",
            f"- Errors: {len(preflight.get('errors', []))}",
            "",
        ]
    if integrity_path.exists():
        integrity = read_json(integrity_path)
        lines += [
            "## Split Integrity",
            "",
            f"- Status: `{integrity.get('status')}`",
            f"- EpiBios: {integrity.get('num_epibios_cases')} cases / {integrity.get('num_epibios_animals')} animals",
            f"- ARAMRA: {integrity.get('num_aramra_cases')} cases / {integrity.get('num_aramra_animals')} animals",
            "- ARAMRA held-out fold animals are not used for selection or fine-tuning within the same fold.",
            "",
        ]
    for title, filename in [
        ("Source-only ARAMRA", "source_only_aramra_summary.csv"),
        ("Main ARAMRA OOF", "aramra_oof_summary.csv"),
        ("EpiBios Retention", "epibios_retention_summary.csv"),
        ("Auxiliary ARAMRA Internal OOF", "aux_aramra_internal_oof_summary.csv"),
    ]:
        path = run_dir / "metrics" / filename
        if not path.exists():
            continue
        lines += [f"## {title}", ""]
        for row in read_csv(path):
            if row.get("group") != "overall":
                continue
            label = " / ".join(str(row.get(key, "")) for key in ["experiment", "method", "budget", "postprocess_variant"] if row.get(key, "") != "")
            lines.append(
                f"- {label}: mean Dice={fmt(row.get('mean_dice'))}, animal macro Dice={fmt(row.get('animal_macro_dice'))}, "
                f"pooled Dice={fmt(row.get('pooled_dice'))}, mean HD95={fmt(row.get('mean_hd95_mm'))}, "
                f"mean ASSD={fmt(row.get('mean_assd_mm'))}, mean absRVE%={fmt(row.get('mean_abs_rve_percent'))}"
            )
        lines.append("")
        time_rows = [row for row in read_csv(path) if str(row.get("group", "")).startswith("time=")]
        if time_rows:
            lines += [f"### {title} Timepoint Subgroups", ""]
            for row in time_rows:
                label = " / ".join(
                    str(row.get(key, ""))
                    for key in ["experiment", "method", "budget", "postprocess_variant", "group"]
                    if row.get(key, "") != ""
                )
                lines.append(
                    f"- {label}: mean Dice={fmt(row.get('mean_dice'))}, "
                    f"animal macro Dice={fmt(row.get('animal_macro_dice'))}, "
                    f"mean HD95={fmt(row.get('mean_hd95_mm'))}, mean ASSD={fmt(row.get('mean_assd_mm'))}"
                )
            lines.append("")
    selected_path = run_dir / "selection" / "selected_animals.csv"
    if selected_path.exists():
        selected_rows = read_csv(selected_path)
        grouped_selection = group_by(selected_rows, ["outer_fold", "method", "budget"])
        lines += ["## Selection Summary", ""]
        for key, rows in sorted(grouped_selection.items()):
            outer_fold, method, budget = key
            m5_count = sum(int(float(row.get("has_m5", 0))) for row in rows)
            cases = sum(int(float(row.get("num_cases", 0))) for row in rows)
            lines.append(
                f"- outer_fold={outer_fold}, method={method}, budget={budget}: "
                f"animals={len(rows)}, cases={cases}, animals_with_M5={m5_count}"
            )
        lines.append("")
    lines += [
        "## Output Map",
        "",
        "- `metadata/`: EpiBios and ARAMRA manifests.",
        "- `splits/`: ARAMRA animal-level folds and source EpiBios fold map.",
        "- `source_predictions/`: workspace_v0 source model ARAMRA predictions used for scoring.",
        "- `scores/`: case-level source disagreement/uncertainty/cost features.",
        "- `selection/`: fold-specific candidate and selected animal tables.",
        "- `checkpoints/`: adapted checkpoints grouped by method, budget, outer fold, and source fold.",
        "- `predictions/`: ARAMRA OOF and EpiBios retention predictions.",
        "- `metrics/`: case-level and summary metrics.",
        "",
        "## Caveats",
        "",
        "- This stage uses existing ARAMRA labels directly; it is not a new manual revision round.",
        "- Disagreement-only selection is intentionally evaluated as a baseline and may be unsafe without human correction.",
        "- Source-only and adapted models must be interpreted separately from the earlier scan-level workspace_v0 OOF results.",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    try:
        number = float(value)
        if math.isnan(number):
            return "NA"
        return f"{number:.6f}"
    except Exception:
        return str(value)


def run_pipeline(cfg: dict[str, Any], run_dir: Path) -> None:
    ensure_run_dirs(run_dir)
    cfg["_run_dir"] = str(run_dir)
    with (run_dir / "config" / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({k: v for k, v in cfg.items() if k != "_run_dir"}, handle, sort_keys=False)
    logger = TeeLogger(run_dir / "logs" / "pipeline.log")
    try:
        update_state(
            run_dir,
            status="running",
            current_stage="initializing",
            started_at=timestamp(),
            pid=os.getpid(),
            command=command_string(),
            git_hash=git_hash(),
            completed_stages=[],
            failed_stage=None,
        )
        seed = int(cfg.get("seed", 20260607))
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.log(f"run_dir={run_dir}")
        logger.log(f"device={select_device(cfg)} torch_cuda_available={torch.cuda.is_available()}")
        preflight_checks(cfg, run_dir, logger)
        epi_cases = aw.scan_epibios_cases(cfg, logger)
        ar_cases = aw.scan_aramra_cases(cfg, logger)
        if not ar_cases:
            raise RuntimeError("No ARAMRA cases were found; cannot run next pipeline")
        update_state(run_dir, current_stage="metadata")
        write_metadata(run_dir, epi_cases, ar_cases, cfg)
        append_completed_stage(run_dir, "metadata")
        score_rows: list[dict[str, Any]] = []
        if bool(cfg["stages"].get("source_prediction_scoring", True)):
            score_rows = predict_source_on_aramra(ar_cases, cfg, run_dir, logger)
        else:
            score_rows = read_csv(run_dir / "scores" / "source_scores_case.csv")
        selection_map = write_selection_tables(run_dir, ar_cases, score_rows, cfg)
        if bool(cfg["stages"].get("run_main_experiments", True)):
            run_main_experiments(epi_cases, ar_cases, score_rows, selection_map, cfg, run_dir, logger)
        run_aux_aramra_internal_oof(ar_cases, cfg, run_dir, logger)
        write_report(run_dir, cfg)
        update_state(run_dir, status="completed", current_stage="completed")
        logger.log("next pipeline completed")
    except Exception as exc:
        update_state(run_dir, status="failed", failed_stage=str(exc))
        logger.log(f"pipeline failed: {exc}")
        raise
    finally:
        logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Next ARAMRA model-guided selection fine-tuning pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--foreground", action="store_true", help="Run in the current process; default launches background worker")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(args.config)
    cfg = load_config(config_path)
    if not args.foreground and not args.worker:
        launch_background(args, config_path, cfg)
        return
    run_dir = make_run_dir(cfg, args.run_dir, args.resume)
    run_pipeline(cfg, run_dir)


if __name__ == "__main__":
    main()
