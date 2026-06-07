from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True), encoding="utf-8")


def load_binary(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    return (np.asarray(image.get_fdata(dtype=np.float32)) > 0).astype(np.uint8)


def load_probability(path: Path) -> np.ndarray:
    payload = np.load(path)
    key = "probability" if "probability" in payload else "s"
    return payload[key].astype(np.float32)


def postprocess(probability: np.ndarray, threshold: float, min_component_voxels: int) -> np.ndarray:
    mask = (probability >= threshold).astype(np.uint8)
    if min_component_voxels <= 1:
        return mask
    labeled, num = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if num == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    keep = np.zeros(num + 1, dtype=bool)
    for idx, size in enumerate(sizes, start=1):
        if int(size) >= min_component_voxels:
            keep[idx] = True
    return keep[labeled].astype(np.uint8)


def dice_from_counts(intersection: int, pred_sum: int, target_sum: int) -> float:
    denom = pred_sum + target_sum
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def surface(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask, ~eroded)


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("nan")
    pred_surface = surface(pred)
    target_surface = surface(target)
    if not pred_surface.any() or not target_surface.any():
        return float("nan")
    dt_pred = ndimage.distance_transform_edt(~pred_surface)
    dt_target = ndimage.distance_transform_edt(~target_surface)
    distances = np.concatenate([dt_target[pred_surface], dt_pred[target_surface]])
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def component_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    pred_labeled, pred_n = ndimage.label(pred, structure=structure)
    target_labeled, target_n = ndimage.label(target, structure=structure)
    pred_hit = 0
    for idx in range(1, pred_n + 1):
        if np.logical_and(pred_labeled == idx, target).any():
            pred_hit += 1
    target_hit = 0
    for idx in range(1, target_n + 1):
        if np.logical_and(target_labeled == idx, pred).any():
            target_hit += 1
    precision = pred_hit / pred_n if pred_n else (1.0 if target_n == 0 else 0.0)
    recall = target_hit / target_n if target_n else (1.0 if pred_n == 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "pred_components": int(pred_n),
        "gt_components": int(target_n),
        "pred_components_hit": int(pred_hit),
        "gt_components_hit": int(target_hit),
        "lesion_precision": float(precision),
        "lesion_recall": float(recall),
        "lesion_f1": float(f1),
    }


def metric_row(meta: dict[str, str], pred: np.ndarray, target: np.ndarray, variant: str, prediction_stage: str, reference_stage: str) -> dict[str, Any]:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    inter = int(np.logical_and(pred_bool, target_bool).sum())
    pred_sum = int(pred_bool.sum())
    target_sum = int(target_bool.sum())
    row = {
        "prediction_stage": prediction_stage,
        "reference_stage": reference_stage,
        "case_id": meta["case_id"],
        "animal_id_strict": meta["animal_id_strict"],
        "animal_family": meta.get("animal_family", ""),
        "time_raw": meta.get("time_raw", ""),
        "timepoint_pattern": meta.get("timepoint_pattern", ""),
        "revised_status": meta.get("revised_status", ""),
        "fold": meta.get("fold", ""),
        "postprocess_variant": variant,
        "dice": dice_from_counts(inter, pred_sum, target_sum),
        "intersection": inter,
        "pred_positive_voxels": pred_sum,
        "gt_positive_voxels": target_sum,
        "fp_voxels": int(np.logical_and(pred_bool, ~target_bool).sum()),
        "fn_voxels": int(np.logical_and(~pred_bool, target_bool).sum()),
        "absolute_volume_error_voxels": abs(pred_sum - target_sum),
        "signed_volume_error_voxels": pred_sum - target_sum,
        "hd95": hd95(pred_bool, target_bool),
    }
    row.update(component_metrics(pred_bool, target_bool))
    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    by_animal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_animal[str(row["animal_id_strict"])].append(row)
    animal_values = [float(np.mean([float(row["dice"]) for row in items])) for items in by_animal.values()]
    inter = sum(int(row["intersection"]) for row in rows)
    pred = sum(int(row["pred_positive_voxels"]) for row in rows)
    gt = sum(int(row["gt_positive_voxels"]) for row in rows)
    hd_values = [float(row["hd95"]) for row in rows if not np.isnan(float(row["hd95"]))]
    return {
        "num_cases": len(rows),
        "num_animals": len(by_animal),
        "macro_dice": float(np.mean([float(row["dice"]) for row in rows])),
        "animal_macro_dice": float(np.mean(animal_values)),
        "micro_dice": dice_from_counts(inter, pred, gt),
        "median_hd95": float(np.median(hd_values)) if hd_values else float("nan"),
        "mean_hd95": float(np.mean(hd_values)) if hd_values else float("nan"),
        "mean_lesion_f1": float(np.mean([float(row["lesion_f1"]) for row in rows])),
        "mean_absolute_volume_error_voxels": float(np.mean([float(row["absolute_volume_error_voxels"]) for row in rows])),
    }


def group_by(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def resolve_label_path(meta: dict[str, str], round_name: str, source_workspace: Optional[Path]) -> Path:
    source_field = "round0_label_path" if round_name == "r0" else "round1_label_path"
    raw_path = meta.get(source_field, "")
    candidates: list[Path] = []
    if raw_path:
        candidates.append(Path(raw_path))
    if source_workspace is not None:
        candidates.append(source_workspace / "artifacts" / "labels" / "binary" / f"round_{0 if round_name == 'r0' else 1}" / f"{meta['case_id']}.nii.gz")
    candidates.extend(
        [
            PROJECT_ROOT / "workspace" / "workspace" / "artifacts" / "labels" / "binary" / f"round_{0 if round_name == 'r0' else 1}" / f"{meta['case_id']}.nii.gz",
            PROJECT_ROOT / "workspace_v0" / "workspace" / "artifacts" / "labels" / "binary" / f"round_{0 if round_name == 'r0' else 1}" / f"{meta['case_id']}.nii.gz",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not resolve {round_name} label for {meta['case_id']}")


def static_reference_analysis(run_dir: Path, source_workspace: Optional[Path], threshold: float, post_values: list[int]) -> None:
    epi_cases = read_csv(run_dir / "metadata" / "epibios_cases.csv")
    out_rows: list[dict[str, Any]] = []
    for idx, meta in enumerate(epi_cases, start=1):
        labels = {
            "r0": load_binary(resolve_label_path(meta, "r0", source_workspace)),
            "r1": load_binary(resolve_label_path(meta, "r1", source_workspace)),
        }
        for pred_stage in ["r0", "r1"]:
            prob_path = run_dir / "oof" / pred_stage / "probabilities" / f"{meta['case_id']}.npz"
            prob = load_probability(prob_path)
            variants = {"raw": (prob >= threshold).astype(np.uint8)}
            for min_voxels in post_values:
                variants[f"post_min{min_voxels}"] = postprocess(prob, threshold, min_voxels)
            for ref_stage, target in labels.items():
                for variant, mask in variants.items():
                    out_rows.append(metric_row(meta, mask, target, variant, pred_stage, ref_stage))
        if idx % 25 == 0:
            print(f"static-reference processed {idx}/{len(epi_cases)} cases", flush=True)

    metrics_dir = ensure_dir(run_dir / "metrics")
    write_csv(metrics_dir / "animalwise_static_reference_case_metrics.csv", out_rows)
    group_rows: list[dict[str, Any]] = []
    grouping_specs = [("overall", None), ("review", "revised_status"), ("family", "animal_family"), ("time", "time_raw"), ("pattern", "timepoint_pattern")]
    for keys, items in group_by(out_rows, ["prediction_stage", "reference_stage", "postprocess_variant"]).items():
        pred_stage, ref_stage, variant = keys
        group_rows.append({"prediction_stage": pred_stage, "reference_stage": ref_stage, "postprocess_variant": variant, "group": "overall", **aggregate_rows(items)})
        for prefix, field in grouping_specs[1:]:
            for value, subitems in group_by(items, [field]).items():
                group_rows.append({"prediction_stage": pred_stage, "reference_stage": ref_stage, "postprocess_variant": variant, "group": f"{prefix}={value[0]}", **aggregate_rows(subitems)})
    write_csv(metrics_dir / "animalwise_static_reference_group_metrics.csv", group_rows)

    summary: dict[str, Any] = {"overall": {}}
    for row in group_rows:
        if row["group"] != "overall":
            continue
        key = f"{row['prediction_stage']}_pred_vs_{row['reference_stage']}_ref_{row['postprocess_variant']}"
        summary["overall"][key] = {k: v for k, v in row.items() if k not in {"prediction_stage", "reference_stage", "postprocess_variant", "group"}}
    write_json(metrics_dir / "animalwise_static_reference_summary.json", summary)


def bootstrap_metric(rows: list[dict[str, Any]], metric: str) -> float:
    if not rows:
        return float("nan")
    if metric == "macro_dice":
        return float(np.mean([float(row["dice"]) for row in rows]))
    if metric == "animal_macro_dice":
        by_animal = group_by(rows, ["animal_id_strict"])
        return float(np.mean([np.mean([float(row["dice"]) for row in items]) for items in by_animal.values()]))
    if metric == "micro_dice":
        inter = sum(int(float(row["intersection"])) for row in rows)
        pred = sum(int(float(row["pred_positive_voxels"])) for row in rows)
        gt = sum(int(float(row["gt_positive_voxels"])) for row in rows)
        return dice_from_counts(inter, pred, gt)
    if metric == "lesion_f1":
        return float(np.mean([float(row["lesion_f1"]) for row in rows]))
    if metric == "median_hd95":
        values = [float(row["hd95"]) for row in rows if row["hd95"] != "" and not np.isnan(float(row["hd95"]))]
        return float(np.median(values)) if values else float("nan")
    if metric == "mean_absolute_volume_error_voxels":
        return float(np.mean([float(row["absolute_volume_error_voxels"]) for row in rows]))
    raise ValueError(metric)


def sample_rows(by_animal: dict[str, list[dict[str, Any]]], sampled_animals: np.ndarray) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    for animal in sampled_animals:
        key = str(animal)
        if key in by_animal:
            sampled.extend(by_animal[key])
        else:
            sampled.extend(by_animal[(key,)])
    return sampled


def aramra_bootstrap(run_dir: Path, n_boot: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    r0_rows = read_csv(run_dir / "metrics" / "aramra_r0_case_metrics.csv")
    r1_rows = read_csv(run_dir / "metrics" / "aramra_r1_case_metrics.csv")
    variants = ["raw", "post_min2"]
    groups = [("overall", None, None), ("time=D9", "time_raw", "D9"), ("time=M5", "time_raw", "M5")]
    metrics = ["macro_dice", "animal_macro_dice", "micro_dice", "lesion_f1", "median_hd95", "mean_absolute_volume_error_voxels"]
    out_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for variant in variants:
        for group_name, group_field, group_value in groups:
            rows_by_stage = {}
            for stage, rows in [("r0", r0_rows), ("r1", r1_rows)]:
                filtered = [row for row in rows if row["postprocess_variant"] == variant]
                if group_field is not None:
                    filtered = [row for row in filtered if row[group_field] == group_value]
                rows_by_stage[stage] = filtered
            animals = sorted(set(row["animal_id_strict"] for row in rows_by_stage["r0"]) & set(row["animal_id_strict"] for row in rows_by_stage["r1"]))
            by_stage_animal = {stage: group_by(rows, ["animal_id_strict"]) for stage, rows in rows_by_stage.items()}
            for metric in metrics:
                observed_r0 = bootstrap_metric(rows_by_stage["r0"], metric)
                observed_r1 = bootstrap_metric(rows_by_stage["r1"], metric)
                deltas = np.zeros(n_boot, dtype=np.float64)
                for idx in range(n_boot):
                    sampled_animals = rng.choice(animals, size=len(animals), replace=True)
                    sampled_r0 = sample_rows(by_stage_animal["r0"], sampled_animals)
                    sampled_r1 = sample_rows(by_stage_animal["r1"], sampled_animals)
                    deltas[idx] = bootstrap_metric(sampled_r1, metric) - bootstrap_metric(sampled_r0, metric)
                ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
                row = {
                    "group": group_name,
                    "postprocess_variant": variant,
                    "metric": metric,
                    "num_animals": len(animals),
                    "num_bootstrap": n_boot,
                    "r0": observed_r0,
                    "r1": observed_r1,
                    "delta_r1_minus_r0": observed_r1 - observed_r0,
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "bootstrap_p_delta_le_0": float(np.mean(deltas <= 0.0)),
                    "bootstrap_p_delta_ge_0": float(np.mean(deltas >= 0.0)),
                }
                out_rows.append(row)
                summary[f"{group_name}_{variant}_{metric}"] = row
    write_csv(run_dir / "metrics" / "aramra_bootstrap_ci.csv", out_rows)
    write_json(run_dir / "metrics" / "aramra_bootstrap_summary.json", summary)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_comparison(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    old_workspace_candidates = [PROJECT_ROOT / "workspace" / "workspace", PROJECT_ROOT / "workspace_v0" / "workspace"]
    old_workspace = next((path for path in old_workspace_candidates if path.exists()), None)
    if old_workspace is not None:
        for round_index in [0, 1]:
            payload = load_json_if_exists(old_workspace / "reports" / f"round_{round_index}" / "oof_summary.json")
            if payload:
                rows.append(
                    {
                        "model_key": f"old_workspace_r{round_index}",
                        "evaluation": "EpiBios scan-level OOF",
                        "postprocess_variant": "raw",
                        "num_cases": payload.get("num_cases"),
                        "num_animals": "",
                        "macro_dice": payload.get("macro_dice_raw"),
                        "animal_macro_dice": "",
                        "micro_dice": payload.get("micro_dice_raw"),
                        "mean_lesion_f1": "",
                        "median_hd95": "",
                        "source_file": str(old_workspace / "reports" / f"round_{round_index}" / "oof_summary.json"),
                    }
                )
                rows.append(
                    {
                        "model_key": f"old_workspace_r{round_index}",
                        "evaluation": "EpiBios scan-level OOF",
                        "postprocess_variant": "postprocessed",
                        "num_cases": payload.get("num_cases"),
                        "num_animals": "",
                        "macro_dice": payload.get("macro_dice_postprocessed"),
                        "animal_macro_dice": "",
                        "micro_dice": payload.get("micro_dice_postprocessed"),
                        "mean_lesion_f1": "",
                        "median_hd95": "",
                        "source_file": str(old_workspace / "reports" / f"round_{round_index}" / "oof_summary.json"),
                    }
                )

    old_aramra_path = PROJECT_ROOT / "analysis" / "workspace_v0_full_external_analysis" / "results" / "aramra_external_group_metrics.csv"
    if old_aramra_path.exists():
        for row in read_csv(old_aramra_path):
            if row.get("time_raw", "") or row.get("animal_family", "") or row.get("animal_timepoint_pattern", ""):
                continue
            if row["postprocess_variant"] not in {"raw", "post_min2"}:
                continue
            rows.append(
                {
                    "model_key": f"old_workspace_r{row['prediction_round']}",
                    "evaluation": "ARAMRA OOD",
                    "postprocess_variant": row["postprocess_variant"],
                    "num_cases": row["num_cases"],
                    "num_animals": row["num_animals"],
                    "macro_dice": row["macro_dice"],
                    "animal_macro_dice": row["animal_macro_dice"],
                    "micro_dice": row["micro_dice"],
                    "mean_lesion_f1": row["mean_lesion_f1"],
                    "median_hd95": row["median_hd95"],
                    "source_file": str(old_aramra_path),
                }
            )

    current_r0 = load_json_if_exists(run_dir / "metrics" / "r0_oof_summary.json").get("overall", {})
    current_r1 = load_json_if_exists(run_dir / "metrics" / "r1_oof_summary.json").get("overall", {})
    for model_key, payload in [("animalwise_r0", current_r0), ("animalwise_r1", current_r1)]:
        for variant in ["raw", "post_min2"]:
            metric = payload.get(variant, {})
            rows.append(
                {
                    "model_key": model_key,
                    "evaluation": "EpiBios animal-wise OOF",
                    "postprocess_variant": variant,
                    "num_cases": metric.get("num_cases"),
                    "num_animals": metric.get("num_animals"),
                    "macro_dice": metric.get("macro_dice"),
                    "animal_macro_dice": metric.get("animal_macro_dice"),
                    "micro_dice": metric.get("micro_dice"),
                    "mean_lesion_f1": metric.get("mean_lesion_f1"),
                    "median_hd95": metric.get("median_hd95"),
                    "source_file": str(run_dir / "metrics" / f"{model_key.split('_')[-1]}_oof_summary.json"),
                }
            )

    aramra_summary = load_json_if_exists(run_dir / "metrics" / "aramra_summary.json").get("overall", {})
    for stage in ["r0", "r1"]:
        for variant in ["raw", "post_min2"]:
            metric = aramra_summary.get(f"{stage}_{variant}", {})
            rows.append(
                {
                    "model_key": f"animalwise_{stage}",
                    "evaluation": "ARAMRA OOD",
                    "postprocess_variant": variant,
                    "num_cases": metric.get("num_cases"),
                    "num_animals": metric.get("num_animals"),
                    "macro_dice": metric.get("macro_dice"),
                    "animal_macro_dice": metric.get("animal_macro_dice"),
                    "micro_dice": metric.get("micro_dice"),
                    "mean_lesion_f1": metric.get("mean_lesion_f1"),
                    "median_hd95": metric.get("median_hd95"),
                    "source_file": str(run_dir / "metrics" / "aramra_summary.json"),
                }
            )
    write_csv(run_dir / "metrics" / "baseline_comparison_table.csv", rows)


def fnum(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def report_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fnum(row.get(col, "")) if col not in {"model_key", "evaluation", "postprocess_variant", "group", "metric", "matrix"} else str(row.get(col, "")) for col in columns) + " |")
    return lines


def write_reports(run_dir: Path) -> None:
    metrics_dir = run_dir / "metrics"
    static_summary = load_json_if_exists(metrics_dir / "animalwise_static_reference_summary.json").get("overall", {})
    static_rows = []
    for key, metric in static_summary.items():
        if key.endswith("_raw") or key.endswith("_post_min2"):
            static_rows.append(
                {
                    "matrix": key,
                    "num_cases": metric.get("num_cases"),
                    "num_animals": metric.get("num_animals"),
                    "macro_dice": metric.get("macro_dice"),
                    "animal_macro_dice": metric.get("animal_macro_dice"),
                    "micro_dice": metric.get("micro_dice"),
                    "mean_lesion_f1": metric.get("mean_lesion_f1"),
                    "median_hd95": metric.get("median_hd95"),
                }
            )
    static_rows = sorted(static_rows, key=lambda row: row["matrix"])
    lines = ["# Static Reference Report", ""]
    lines += report_table(static_rows, ["matrix", "num_cases", "num_animals", "macro_dice", "animal_macro_dice", "micro_dice", "mean_lesion_f1", "median_hd95"])
    lines += ["", "Detailed subgroup metrics are in `metrics/animalwise_static_reference_group_metrics.csv`.", ""]
    (run_dir / "static_reference_report.md").write_text("\n".join(lines), encoding="utf-8")

    bootstrap_rows = read_csv(metrics_dir / "aramra_bootstrap_ci.csv")
    keep = [row for row in bootstrap_rows if row["metric"] in {"macro_dice", "animal_macro_dice", "micro_dice", "lesion_f1"}]
    lines = ["# ARAMRA Bootstrap CI Report", "", f"- Bootstrap repeats: `{bootstrap_rows[0]['num_bootstrap'] if bootstrap_rows else ''}`", "- Unit: `animal_id_strict`", ""]
    lines += report_table(keep, ["group", "postprocess_variant", "metric", "num_animals", "r0", "r1", "delta_r1_minus_r0", "ci95_low", "ci95_high", "bootstrap_p_delta_le_0"])
    lines += ["", "Full metric output, including HD95 and volume error, is in `metrics/aramra_bootstrap_ci.csv`.", ""]
    (run_dir / "aramra_bootstrap_report.md").write_text("\n".join(lines), encoding="utf-8")

    baseline_rows = read_csv(metrics_dir / "baseline_comparison_table.csv")
    lines = ["# Baseline Comparison Report", ""]
    lines += report_table(baseline_rows, ["model_key", "evaluation", "postprocess_variant", "num_cases", "num_animals", "macro_dice", "animal_macro_dice", "micro_dice", "mean_lesion_f1", "median_hd95"])
    lines += ["", "This table separates old scan-level OOF, current animal-wise OOF, and ARAMRA OOD evaluations.", ""]
    (run_dir / "baseline_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-workspace", default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    source_workspace = Path(args.source_workspace).resolve() if args.source_workspace else None
    post_values = [2, 16]
    threshold = 0.5

    print("running static-reference analysis", flush=True)
    static_reference_analysis(run_dir, source_workspace, threshold, post_values)
    print("running ARAMRA bootstrap", flush=True)
    aramra_bootstrap(run_dir, args.bootstrap, args.seed)
    print("running baseline comparison", flush=True)
    baseline_comparison(run_dir)
    write_reports(run_dir)
    print(f"completed reports in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
