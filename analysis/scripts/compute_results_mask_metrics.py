from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results"
RUN_DIR = PROJECT_ROOT / "analysis" / "animalwise_oof_pipeline" / "runs" / "20260524_223407_server_animalwise_oof"
OLD_ARAMRA_ROOT = PROJECT_ROOT / "analysis" / "workspace_v0_full_external_analysis" / "predictions" / "aramra"
EPI_LABEL_ROOT = PROJECT_ROOT / "workspace" / "workspace" / "artifacts" / "labels" / "binary"
ARAMRA_LABEL_ROOT = Path("E:/Hemorrhage/ARAMRA002_standardized/labelsTs")
OUT_DIR = RESULTS_ROOT / "metrics_full"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def case_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
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
        for row in rows:
            writer.writerow({key: serialise_cell(row.get(key)) for key in fieldnames})


def serialise_cell(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return value
    if isinstance(value, (np.integer, np.floating)):
        return serialise_cell(float(value))
    return value


def write_json(path: Path, payload: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, (np.integer, np.floating)):
            return clean(float(value))
        return value

    ensure_dir(path.parent)
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True), encoding="utf-8")


def load_metadata() -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    for path in [RUN_DIR / "metadata" / "epibios_cases.csv", RUN_DIR / "metadata" / "aramra_cases.csv"]:
        for row in read_csv(path):
            meta[row["case_id"]] = row
    return meta


def round_name(folder_name: str) -> str:
    if folder_name == "round_0":
        return "r0"
    if folder_name == "round_1":
        return "r1"
    raise ValueError(f"Unexpected round folder: {folder_name}")


def variant_name(folder_name: str) -> str:
    if folder_name == "masks_raw":
        return "raw"
    if folder_name == "masks_post_min2":
        return "post_min2"
    if folder_name == "masks_post_min16":
        return "post_min16"
    return folder_name


def resolve_reference(case_id: str, cohort: str, reference_round: str) -> Path:
    if cohort == "EpiBios":
        round_index = 0 if reference_round == "r0" else 1
        path = EPI_LABEL_ROOT / f"round_{round_index}" / f"{case_id}.nii.gz"
    elif cohort == "ARAMRA002":
        path = ARAMRA_LABEL_ROOT / f"{case_id}.nii.gz"
    else:
        raise ValueError(f"Unexpected cohort: {cohort}")
    if not path.exists():
        raise FileNotFoundError(f"Missing reference for {case_id}: {path}")
    return path


def prediction_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    # Animal-level EpiBios OOF.
    base = RESULTS_ROOT / "animal-level" / "epibios oof"
    for round_dir in sorted(base.glob("round_*")):
        if not round_dir.is_dir():
            continue
        for variant_dir in sorted(round_dir.glob("masks_*")):
            for pred_path in sorted(variant_dir.glob("*.nii.gz")):
                records.append(
                    {
                        "analysis_level": "animal-level",
                        "cohort": "EpiBios",
                        "evaluation": "epibios_oof",
                        "prediction_round": round_name(round_dir.name),
                        "postprocess_variant": variant_name(variant_dir.name),
                        "case_id": case_id_from_path(pred_path),
                        "prediction_path": str(pred_path.resolve()),
                    }
                )

    # Animal-level ARAMRA predictions.
    base = RESULTS_ROOT / "animal-level" / "aramra predictions"
    for round_dir in sorted(base.glob("round_*")):
        if not round_dir.is_dir():
            continue
        for variant_dir in sorted(round_dir.glob("masks_*")):
            for pred_path in sorted(variant_dir.glob("*.nii.gz")):
                records.append(
                    {
                        "analysis_level": "animal-level",
                        "cohort": "ARAMRA002",
                        "evaluation": "aramra_ood",
                        "prediction_round": round_name(round_dir.name),
                        "postprocess_variant": variant_name(variant_dir.name),
                        "case_id": case_id_from_path(pred_path),
                        "prediction_path": str(pred_path.resolve()),
                    }
                )

    # Case-level EpiBios OOF copied from server; only one provided mask variant.
    base = RESULTS_ROOT / "case-level"
    for round_dir in sorted(base.glob("round_*")):
        if not round_dir.is_dir():
            continue
        for pred_path in sorted(round_dir.glob("*.nii.gz")):
            records.append(
                {
                    "analysis_level": "case-level",
                    "cohort": "EpiBios",
                    "evaluation": "epibios_oof",
                    "prediction_round": round_name(round_dir.name),
                    "postprocess_variant": "provided",
                    "case_id": case_id_from_path(pred_path),
                    "prediction_path": str(pred_path.resolve()),
                }
            )

    # Case-level ARAMRA OOD is available in the local workspace_v0 external analysis.
    base = OLD_ARAMRA_ROOT
    if base.exists():
        for round_dir in sorted(base.glob("round_*")):
            if not round_dir.is_dir():
                continue
            for variant_dir in sorted(round_dir.glob("masks_*")):
                for pred_path in sorted(variant_dir.glob("*.nii.gz")):
                    records.append(
                        {
                            "analysis_level": "case-level",
                            "cohort": "ARAMRA002",
                            "evaluation": "aramra_ood",
                            "prediction_round": round_name(round_dir.name),
                            "postprocess_variant": variant_name(variant_dir.name),
                            "case_id": case_id_from_path(pred_path),
                            "prediction_path": str(pred_path.resolve()),
                        }
                    )

    return records


def load_mask(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32)) > 0
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    return data, spacing


def binary_surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask, ~eroded)


def surface_distances(pred: np.ndarray, ref: np.ndarray, spacing: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
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


def mask_metrics(pred: np.ndarray, ref: np.ndarray, spacing: Tuple[float, float, float]) -> Dict[str, Any]:
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, ref {ref.shape}")
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
    percent_agreement = safe_div(tp + tn, total, float("nan"))

    pred_rate = safe_div(pred_voxels, total, 0.0)
    ref_rate = safe_div(ref_voxels, total, 0.0)
    expected_agreement = pred_rate * ref_rate + (1.0 - pred_rate) * (1.0 - ref_rate)
    kappa = safe_div(percent_agreement - expected_agreement, 1.0 - expected_agreement, float("nan"))

    if not pred.any() and not ref.any():
        hd95 = 0.0
        assd = 0.0
        surface_dice_1mm = 1.0
    elif not pred.any() or not ref.any():
        hd95 = float("nan")
        assd = float("nan")
        surface_dice_1mm = 0.0
    else:
        d_pred_to_ref, d_ref_to_pred = surface_distances(pred, ref, spacing)
        distances = np.concatenate([d_pred_to_ref, d_ref_to_pred])
        hd95 = float(np.percentile(distances, 95)) if distances.size else float("nan")
        assd = float(np.mean(distances)) if distances.size else float("nan")
        within = int((d_pred_to_ref <= 1.0).sum() + (d_ref_to_pred <= 1.0).sum())
        denom = int(d_pred_to_ref.size + d_ref_to_pred.size)
        surface_dice_1mm = safe_div(within, denom, float("nan"))

    return {
        "dice": float(dice),
        "jaccard": float(jaccard),
        "hd95_mm": hd95,
        "assd_mm": assd,
        "rve": rve,
        "rve_percent": rve * 100.0 if not math.isnan(rve) else float("nan"),
        "abs_rve": abs_rve,
        "abs_rve_percent": abs_rve * 100.0 if not math.isnan(abs_rve) else float("nan"),
        "volume_similarity": float(volume_similarity),
        "surface_dice_1mm": float(surface_dice_1mm),
        "voxel_agreement": float(percent_agreement),
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


class MaskCache:
    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[np.ndarray, Tuple[float, float, float]]] = {}

    def get(self, path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        key = str(path.resolve())
        if key not in self._cache:
            self._cache[key] = load_mask(path)
        return self._cache[key]


def enrich_record(record: Dict[str, Any], metadata: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    row = dict(record)
    meta = metadata.get(str(record["case_id"]), {})
    for key in [
        "animal_id_strict",
        "animal_id_raw",
        "animal_family",
        "time_raw",
        "timepoint_pattern",
        "fold",
        "revised_status",
    ]:
        row[key] = meta.get(key, "")
    return row


def compute_case_metrics(records: List[Dict[str, Any]], metadata: Dict[str, Dict[str, str]], cache: MaskCache) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        case_id = str(record["case_id"])
        reference_round = str(record["prediction_round"]) if record["cohort"] == "EpiBios" else "eval"
        reference_path = resolve_reference(case_id, str(record["cohort"]), reference_round)
        pred, spacing = cache.get(Path(str(record["prediction_path"])))
        ref, ref_spacing = cache.get(reference_path)
        metric = mask_metrics(pred, ref, spacing)
        row = enrich_record(record, metadata)
        row.update(
            {
                "comparison_type": "prediction_vs_round_specific_reference",
                "reference_round": reference_round,
                "reference_path": str(reference_path.resolve()),
                "spacing_x_mm": spacing[0],
                "spacing_y_mm": spacing[1],
                "spacing_z_mm": spacing[2],
                "reference_spacing_x_mm": ref_spacing[0],
                "reference_spacing_y_mm": ref_spacing[1],
                "reference_spacing_z_mm": ref_spacing[2],
            }
        )
        row.update(metric)
        rows.append(row)
        if idx % 250 == 0:
            print(f"case metrics {idx}/{len(records)}", flush=True)
    return rows


def compute_static_reference_metrics(records: List[Dict[str, Any]], metadata: Dict[str, Dict[str, str]], cache: MaskCache) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    epi_records = [record for record in records if record["cohort"] == "EpiBios"]
    total = len(epi_records) * 2
    done = 0
    for record in epi_records:
        case_id = str(record["case_id"])
        pred, spacing = cache.get(Path(str(record["prediction_path"])))
        for reference_round in ["r0", "r1"]:
            reference_path = resolve_reference(case_id, "EpiBios", reference_round)
            ref, _ = cache.get(reference_path)
            metric = mask_metrics(pred, ref, spacing)
            row = enrich_record(record, metadata)
            row.update(
                {
                    "comparison_type": "prediction_vs_static_reference",
                    "reference_round": reference_round,
                    "reference_path": str(reference_path.resolve()),
                }
            )
            row.update(metric)
            rows.append(row)
            done += 1
            if done % 250 == 0:
                print(f"static reference {done}/{total}", flush=True)
    return rows


def compute_prediction_agreement(records: List[Dict[str, Any]], metadata: Dict[str, Dict[str, str]], cache: MaskCache) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str, str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            str(record["analysis_level"]),
            str(record["cohort"]),
            str(record["evaluation"]),
            str(record["postprocess_variant"]),
            str(record["case_id"]),
        )
        grouped[key][str(record["prediction_round"])] = record

    for key, by_round in sorted(grouped.items()):
        if "r0" not in by_round or "r1" not in by_round:
            continue
        r0 = by_round["r0"]
        r1 = by_round["r1"]
        r0_mask, spacing = cache.get(Path(str(r0["prediction_path"])))
        r1_mask, _ = cache.get(Path(str(r1["prediction_path"])))
        metric = mask_metrics(r1_mask, r0_mask, spacing)
        base = enrich_record(r1, metadata)
        base.update(
            {
                "comparison_type": "prediction_round_agreement",
                "reference_is": "r0_prediction",
                "moving_is": "r1_prediction",
                "r0_prediction_path": str(Path(str(r0["prediction_path"])).resolve()),
                "r1_prediction_path": str(Path(str(r1["prediction_path"])).resolve()),
            }
        )
        base.update(metric)
        rows.append(base)
    return rows


def compute_label_agreement(metadata: Dict[str, Dict[str, str]], cache: MaskCache) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case_id, meta in sorted(metadata.items()):
        if meta.get("cohort") != "EpiBios":
            continue
        r0_path = resolve_reference(case_id, "EpiBios", "r0")
        r1_path = resolve_reference(case_id, "EpiBios", "r1")
        r0_label, spacing = cache.get(r0_path)
        r1_label, _ = cache.get(r1_path)
        metric = mask_metrics(r1_label, r0_label, spacing)
        row = {
            "comparison_type": "human_label_round_agreement",
            "analysis_level": "label",
            "cohort": "EpiBios",
            "evaluation": "label_shift",
            "prediction_round": "r1_label",
            "postprocess_variant": "label",
            "case_id": case_id,
            "reference_round": "r0_label",
            "animal_id_strict": meta.get("animal_id_strict", ""),
            "animal_id_raw": meta.get("animal_id_raw", ""),
            "animal_family": meta.get("animal_family", ""),
            "time_raw": meta.get("time_raw", ""),
            "timepoint_pattern": meta.get("timepoint_pattern", ""),
            "fold": meta.get("fold", ""),
            "revised_status": meta.get("revised_status", ""),
            "r0_label_path": str(r0_path.resolve()),
            "r1_label_path": str(r1_path.resolve()),
        }
        row.update(metric)
        rows.append(row)
    return rows


def aggregate(rows: List[Dict[str, Any]], keys: Iterable[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    keys = list(keys)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)

    out: List[Dict[str, Any]] = []
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
        "absolute_volume_error_mm3",
    ]
    for group_key, items in sorted(grouped.items()):
        row = {key: value for key, value in zip(keys, group_key)}
        row["num_cases"] = len(items)
        animals = {str(item.get("animal_id_strict", "")) for item in items if item.get("animal_id_strict", "")}
        row["num_animals"] = len(animals)
        inter = sum(int(item.get("tp_voxels", 0)) for item in items)
        pred = sum(int(item.get("pred_positive_voxels", 0)) for item in items)
        ref = sum(int(item.get("ref_positive_voxels", 0)) for item in items)
        row["pooled_dice"] = 1.0 if pred + ref == 0 else 2.0 * inter / (pred + ref)
        for metric in metric_names:
            values = [float(item[metric]) for item in items if item.get(metric, "") != "" and not math.isnan(float(item[metric]))]
            row[f"mean_{metric}"] = float(np.mean(values)) if values else float("nan")
            row[f"median_{metric}"] = float(np.median(values)) if values else float("nan")
        out.append(row)
    return out


def round_delta(case_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in case_rows:
        key = (
            str(row["analysis_level"]),
            str(row["cohort"]),
            str(row["evaluation"]),
            str(row["postprocess_variant"]),
            str(row["case_id"]),
        )
        grouped[key][str(row["prediction_round"])] = row
    out: List[Dict[str, Any]] = []
    for _, by_round in sorted(grouped.items()):
        if "r0" not in by_round or "r1" not in by_round:
            continue
        r0 = by_round["r0"]
        r1 = by_round["r1"]
        row = {
            "analysis_level": r1["analysis_level"],
            "cohort": r1["cohort"],
            "evaluation": r1["evaluation"],
            "postprocess_variant": r1["postprocess_variant"],
            "case_id": r1["case_id"],
            "animal_id_strict": r1.get("animal_id_strict", ""),
            "animal_family": r1.get("animal_family", ""),
            "time_raw": r1.get("time_raw", ""),
            "timepoint_pattern": r1.get("timepoint_pattern", ""),
            "r0_dice": r0.get("dice", ""),
            "r1_dice": r1.get("dice", ""),
            "delta_dice_r1_minus_r0": float(r1["dice"]) - float(r0["dice"]),
            "r0_hd95_mm": r0.get("hd95_mm", ""),
            "r1_hd95_mm": r1.get("hd95_mm", ""),
            "delta_hd95_r1_minus_r0": float(r1["hd95_mm"]) - float(r0["hd95_mm"])
            if r0.get("hd95_mm", "") != "" and r1.get("hd95_mm", "") != "" and not math.isnan(float(r0["hd95_mm"])) and not math.isnan(float(r1["hd95_mm"]))
            else float("nan"),
            "r0_assd_mm": r0.get("assd_mm", ""),
            "r1_assd_mm": r1.get("assd_mm", ""),
            "delta_assd_r1_minus_r0": float(r1["assd_mm"]) - float(r0["assd_mm"])
            if r0.get("assd_mm", "") != "" and r1.get("assd_mm", "") != "" and not math.isnan(float(r0["assd_mm"])) and not math.isnan(float(r1["assd_mm"]))
            else float("nan"),
            "r0_abs_rve_percent": r0.get("abs_rve_percent", ""),
            "r1_abs_rve_percent": r1.get("abs_rve_percent", ""),
            "delta_abs_rve_percent_r1_minus_r0": float(r1["abs_rve_percent"]) - float(r0["abs_rve_percent"])
            if r0.get("abs_rve_percent", "") != "" and r1.get("abs_rve_percent", "") != "" and not math.isnan(float(r0["abs_rve_percent"])) and not math.isnan(float(r1["abs_rve_percent"]))
            else float("nan"),
        }
        out.append(row)
    return out


def main() -> None:
    ensure_dir(OUT_DIR)
    metadata = load_metadata()
    records = prediction_records()
    cache = MaskCache()

    manifest_rows = [enrich_record(record, metadata) for record in records]
    for row in manifest_rows:
        if row["cohort"] == "EpiBios":
            row["round_specific_reference_path"] = str(resolve_reference(row["case_id"], "EpiBios", row["prediction_round"]).resolve())
        else:
            row["round_specific_reference_path"] = str(resolve_reference(row["case_id"], "ARAMRA002", "eval").resolve())

    write_csv(OUT_DIR / "prediction_manifest.csv", manifest_rows)
    print(f"manifest rows: {len(manifest_rows)}", flush=True)

    case_rows = compute_case_metrics(records, metadata, cache)
    write_csv(OUT_DIR / "mask_case_metrics.csv", case_rows)
    print(f"case metric rows: {len(case_rows)}", flush=True)

    static_rows = compute_static_reference_metrics(records, metadata, cache)
    write_csv(OUT_DIR / "epibios_static_reference_metrics.csv", static_rows)
    print(f"static reference rows: {len(static_rows)}", flush=True)

    pred_agreement_rows = compute_prediction_agreement(records, metadata, cache)
    write_csv(OUT_DIR / "prediction_round_agreement_metrics.csv", pred_agreement_rows)
    print(f"prediction agreement rows: {len(pred_agreement_rows)}", flush=True)

    label_agreement_rows = compute_label_agreement(metadata, cache)
    write_csv(OUT_DIR / "epibios_label_round_agreement_metrics.csv", label_agreement_rows)
    print(f"label agreement rows: {len(label_agreement_rows)}", flush=True)

    deltas = round_delta(case_rows)
    write_csv(OUT_DIR / "round_delta_case_metrics.csv", deltas)

    summary_keys = ["analysis_level", "cohort", "evaluation", "prediction_round", "postprocess_variant"]
    overall_summary = aggregate(case_rows, summary_keys)
    time_summary = aggregate(case_rows, summary_keys + ["time_raw"])
    pattern_summary = aggregate(case_rows, summary_keys + ["timepoint_pattern"])
    static_summary = aggregate(static_rows, ["analysis_level", "cohort", "evaluation", "prediction_round", "reference_round", "postprocess_variant"])
    agreement_summary = aggregate(pred_agreement_rows, ["analysis_level", "cohort", "evaluation", "postprocess_variant"])
    label_summary = aggregate(label_agreement_rows, ["cohort", "evaluation", "postprocess_variant"])

    write_csv(OUT_DIR / "summary_overall.csv", overall_summary)
    write_csv(OUT_DIR / "summary_by_timepoint.csv", time_summary)
    write_csv(OUT_DIR / "summary_by_timepoint_pattern.csv", pattern_summary)
    write_csv(OUT_DIR / "summary_static_reference.csv", static_summary)
    write_csv(OUT_DIR / "summary_prediction_agreement.csv", agreement_summary)
    write_csv(OUT_DIR / "summary_label_agreement.csv", label_summary)

    integrity = {
        "results_root": str(RESULTS_ROOT.resolve()),
        "old_case_level_aramra_predictions_used": str(OLD_ARAMRA_ROOT.resolve()),
        "old_case_level_aramra_predictions_exists": OLD_ARAMRA_ROOT.exists(),
        "epibios_label_root": str(EPI_LABEL_ROOT.resolve()),
        "aramra_label_root": str(ARAMRA_LABEL_ROOT.resolve()),
        "prediction_records": len(records),
        "case_metric_rows": len(case_rows),
        "static_reference_rows": len(static_rows),
        "prediction_agreement_rows": len(pred_agreement_rows),
        "label_agreement_rows": len(label_agreement_rows),
        "outputs": sorted(path.name for path in OUT_DIR.glob("*.csv")),
    }
    write_json(OUT_DIR / "metrics_integrity_report.json", integrity)
    print(json.dumps(integrity, indent=2), flush=True)


if __name__ == "__main__":
    main()

