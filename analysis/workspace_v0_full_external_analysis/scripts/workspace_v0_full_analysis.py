from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(DEFAULT_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT_ROOT / "src"))

from hemorrhage.data.nifti import percentile_zscore  # noqa: E402
from hemorrhage.training.inference import apply_tta, invert_tta, sliding_window_predict  # noqa: E402
from hemorrhage.training.model import ResidualUNet3D  # noqa: E402
from hemorrhage.training.postprocess import postprocess_probability_map  # noqa: E402


TIME_NUMERIC = {
    "D02": 2.0,
    "D09": 9.0,
    "D9": 9.0,
    "D28": 28.0,
    "W04": 28.0,
    "M01": 30.0,
    "M5": 150.0,
    "M05": 150.0,
    "W20": 140.0,
}
TIME_ORDER = {"D02": 2, "D09": 9, "D9": 9, "D28": 28, "W04": 28, "M01": 30, "M05": 150, "M5": 150, "W20": 140}
SESSION_ORDER = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}
TTA_MODES = ["identity", "flip_x", "flip_y", "flip_xy"]
PATCH_SIZE = (128, 96, 64)
OVERLAP = 0.5
THRESHOLD = 0.5
BOOTSTRAP_SEED = 20260523


@dataclass(frozen=True)
class Paths:
    project_root: Path
    out_dir: Path
    workspace_v0: Path
    data_root: Path
    aramra_root: Path


def make_paths(project_root: Path, out_dir: Path, aramra_root: Path) -> Paths:
    project_root = project_root.resolve()
    return Paths(
        project_root=project_root,
        out_dir=out_dir.resolve(),
        workspace_v0=(project_root / "workspace_v0" / "workspace").resolve(),
        data_root=(project_root / "data").resolve(),
        aramra_root=aramra_root.resolve(),
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    last_error: PermissionError | None = None
    for _ in range(50):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1)
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        if tmp.exists():
            tmp.unlink()
    if last_error is not None and not path.exists():
        raise last_error


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, tuple, dict)):
                    clean[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                elif value is None:
                    clean[key] = ""
                else:
                    clean[key] = value
            writer.writerow(clean)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def strip_nii_gz(name: str) -> str:
    if not name.endswith(".nii.gz"):
        raise ValueError(f"Expected .nii.gz: {name}")
    return name[: -len(".nii.gz")]


def image_case_id(path: Path) -> str:
    stem = strip_nii_gz(path.name)
    if not stem.endswith("_0000"):
        raise ValueError(f"Project image does not end with _0000: {path}")
    return stem[: -len("_0000")]


def load_nifti(path: Path, dtype: np.dtype = np.float32) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=dtype)
    return data, img.affine.copy(), img.header.copy()


def load_label_bool(path: Path) -> np.ndarray:
    data, _, _ = load_nifti(path, dtype=np.float32)
    return (data > 0).astype(np.uint8)


def save_nifti_like(path: Path, data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, dtype: np.dtype = np.uint8) -> None:
    ensure_dir(path.parent)
    out = nib.Nifti1Image(data.astype(dtype), affine, header.copy())
    out.set_data_dtype(dtype)
    nib.save(out, str(path))


def nifti_geometry(path: Path) -> dict[str, Any]:
    img = nib.load(str(path))
    return {
        "shape": "x".join(str(int(v)) for v in img.shape[:3]),
        "spacing": ",".join(f"{float(v):.6g}" for v in img.header.get_zooms()[:3]),
        "orientation": "".join(str(v) for v in nib.aff2axcodes(img.affine)),
    }


def label_histogram(path: Path) -> dict[str, int]:
    data, _, _ = load_nifti(path, dtype=np.float32)
    values, counts = np.unique(data.astype(np.int16), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def parse_date_prefix(case_id: str) -> str:
    match = re.match(r"^(\d{8})_", case_id)
    return match.group(1) if match else ""


def parse_epibios_case(case_id: str) -> dict[str, Any]:
    animal_match = re.search(r"(MHR_\d+|Rat\d+)", case_id)
    animal_raw = animal_match.group(1) if animal_match else case_id
    family = "MHR" if animal_raw.startswith("MHR_") else ("B4C_Rat" if animal_raw.startswith("Rat") else "unknown")
    explicit = re.search(r"_(D02|D09|D28|M01|M05|W04|W20)(?:_|$)", case_id)
    if explicit:
        time_raw = explicit.group(1)
        time_class = "calendar"
        session_ordinal = ""
    else:
        session = re.search(r"_M_1_([1-4])_", case_id)
        time_raw = f"S{session.group(1)}" if session else "unknown"
        time_class = "ordinal_session" if session else "unknown"
        session_ordinal = int(session.group(1)) if session else ""
    time_numeric = TIME_NUMERIC.get(time_raw, "")
    return {
        "animal_id_raw": animal_raw,
        "animal_id_strict": animal_raw,
        "animal_family": family,
        "time_raw": time_raw,
        "time_numeric": time_numeric,
        "time_class": time_class,
        "session_ordinal": session_ordinal,
        "scan_date": parse_date_prefix(case_id),
    }


def normalize_aramra_case_id(path: Path) -> tuple[str, str]:
    stem = strip_nii_gz(path.name)
    if stem.endswith("_meanMag"):
        return stem[: -len("_meanMag")], "image"
    if stem.endswith("_mag_SEG"):
        return stem[: -len("_mag_SEG")], "label"
    if stem.endswith("_SEG"):
        return stem[: -len("_SEG")], "label"
    raise ValueError(f"Cannot classify ARAMRA file name: {path.name}")


def parse_aramra_case(case_id: str) -> dict[str, Any]:
    match = re.search(r"ARAMRA002_([^_]+)_(D9|9D|M5|5M)(?:_|$)", case_id, flags=re.IGNORECASE)
    animal_raw = match.group(1) if match else ""
    time_token = match.group(2).upper() if match else ""
    time_raw = "D9" if time_token in {"D9", "9D"} else ("M5" if time_token in {"M5", "5M"} else "unknown")
    raw_no_prefix = re.sub(r"^R", "", animal_raw, flags=re.IGNORECASE)
    digit_match = re.match(r"(\d+)", raw_no_prefix)
    animal_strict = f"A{digit_match.group(1)}" if digit_match else animal_raw
    return {
        "animal_id_raw": animal_raw,
        "animal_id_strict": animal_strict,
        "animal_family": "ARAMRA002",
        "time_raw": time_raw,
        "time_numeric": TIME_NUMERIC.get(time_raw, ""),
        "time_class": "calendar",
        "session_ordinal": "",
        "scan_date": parse_date_prefix(case_id),
    }


def session_sort_key(row: dict[str, Any]) -> tuple[float, str, str]:
    if row.get("time_numeric") != "":
        try:
            primary = float(row["time_numeric"])
        except (TypeError, ValueError):
            primary = float("inf")
    elif row.get("session_ordinal") != "":
        primary = 1000.0 + float(row["session_ordinal"])
    else:
        primary = float("inf")
    return primary, str(row.get("scan_date", "")), str(row.get("case_id", ""))


def localized_workspace_path(paths: Paths, server_path: str | None) -> str:
    if not server_path:
        return ""
    raw = str(server_path).replace("\\", "/")
    marker = "/workspace/"
    if marker in raw:
        suffix = raw.split(marker, 1)[1]
        return str((paths.workspace_v0 / suffix).resolve())
    marker = "/data/"
    if marker in raw:
        suffix = raw.split(marker, 1)[1]
        return str((paths.data_root / suffix).resolve())
    return server_path


def read_workspace_db(paths: Paths) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    db_path = paths.workspace_v0 / "state.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cases: dict[str, dict[str, Any]] = {}
    for row in conn.execute("SELECT * FROM cases ORDER BY case_id"):
        item = dict(row)
        item["image_path"] = localized_workspace_path(paths, item.get("image_path"))
        item["source_label_path"] = localized_workspace_path(paths, item.get("source_label_path"))
        item["current_raw_label_path"] = localized_workspace_path(paths, item.get("current_raw_label_path"))
        item["current_binary_label_path"] = localized_workspace_path(paths, item.get("current_binary_label_path"))
        item["metadata"] = json.loads(item.get("metadata_json") or "{}")
        cases[item["case_id"]] = item
    review_stats: dict[str, dict[str, Any]] = {}
    for row in conn.execute("SELECT * FROM review_stats WHERE round_index = 1 ORDER BY case_id"):
        item = dict(row)
        for key in ["routine_final_label_path", "audit_anchor_label_path", "audit_final_label_path"]:
            item[key] = localized_workspace_path(paths, item.get(key))
        review_stats[item["case_id"]] = item
    rounds: dict[str, Any] = {}
    for row in conn.execute("SELECT * FROM rounds ORDER BY round_index"):
        item = dict(row)
        rounds[str(item["round_index"])] = {
            "status": item["status"],
            "budget": item["budget"],
            "metrics": json.loads(item.get("metrics_json") or "{}"),
            "routine_ids": json.loads(item.get("routine_ids_json") or "null"),
            "audit_ids": json.loads(item.get("audit_ids_json") or "null"),
            "progress": json.loads(item.get("progress_json") or "null"),
        }
    return cases, review_stats, rounds


def scan_epibios(paths: Paths, workspace_cases: dict[str, dict[str, Any]], review_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    images = sorted((paths.data_root / "imagesTr").glob("*.nii.gz"))
    labels_dir = paths.data_root / "labelsTr"
    rows: list[dict[str, Any]] = []
    for image_path in images:
        case_id = image_case_id(image_path)
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        parsed = parse_epibios_case(case_id)
        db = workspace_cases.get(case_id, {})
        review = review_stats.get(case_id, {})
        role = str(review.get("role") or "none")
        image_geo = nifti_geometry(image_path)
        label_geo = nifti_geometry(label_path)
        hist = label_histogram(label_path)
        row = {
            "case_id": case_id,
            "cohort": "EpiBios",
            "image_path": str(image_path.resolve()),
            "label_path": str(label_path.resolve()),
            "has_label": True,
            "has_image": True,
            "label_round": "round1_current" if (paths.workspace_v0 / "artifacts" / "labels" / "binary" / "round_1" / f"{case_id}.nii.gz").exists() else "round0_only",
            "label_projection": "codes_1_3_foreground",
            "field_strength": "unknown_epibios_mixed",
            "field_strength_known": False,
            "original_fold": int(db.get("fold_id", -1)) if db else "",
            "v0_positive_voxels": int(db.get("v0", -1)) if db else "",
            "revised_status": role,
            "label_trust": "high_revised" if role in {"routine", "audit"} else "medium_unreviewed",
            "split_new": "source_operational_hitl",
            "image_shape": image_geo["shape"],
            "label_shape": label_geo["shape"],
            "image_spacing": image_geo["spacing"],
            "label_spacing": label_geo["spacing"],
            "image_orientation": image_geo["orientation"],
            "label_orientation": label_geo["orientation"],
            "shape_match": image_geo["shape"] == label_geo["shape"],
            "spacing_match": image_geo["spacing"] == label_geo["spacing"],
            "orientation_match": image_geo["orientation"] == label_geo["orientation"],
            "label_histogram": hist,
        }
        row.update(parsed)
        rows.append(row)
    return rows


def scan_aramra(paths: Paths) -> list[dict[str, Any]]:
    case_map: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(paths.aramra_root.rglob("*ARAMRA*.nii.gz")):
        if path.name.startswith("._"):
            continue
        try:
            case_id, kind = normalize_aramra_case_id(path)
        except ValueError:
            continue
        if kind in case_map[case_id]:
            raise RuntimeError(f"Duplicate ARAMRA {kind} for {case_id}: {case_map[case_id][kind]} and {path}")
        case_map[case_id][kind] = path
    rows: list[dict[str, Any]] = []
    for case_id in sorted(case_map):
        image_path = case_map[case_id].get("image")
        label_path = case_map[case_id].get("label")
        parsed = parse_aramra_case(case_id)
        image_geo = nifti_geometry(image_path) if image_path else {}
        label_geo = nifti_geometry(label_path) if label_path else {}
        hist = label_histogram(label_path) if label_path else {}
        row = {
            "case_id": case_id,
            "cohort": "ARAMRA002",
            "image_path": str(image_path.resolve()) if image_path else "",
            "label_path": str(label_path.resolve()) if label_path else "",
            "has_label": bool(label_path),
            "has_image": bool(image_path),
            "label_round": "external_initial_label" if label_path else "unlabeled",
            "label_projection": "nonzero_foreground",
            "field_strength": "9.4T",
            "field_strength_known": True,
            "original_fold": "",
            "v0_positive_voxels": "",
            "revised_status": "none",
            "label_trust": "medium_unknown_unreviewed" if label_path else "unlabeled",
            "split_new": "locked_external_evaluation_current",
            "image_shape": image_geo.get("shape", ""),
            "label_shape": label_geo.get("shape", ""),
            "image_spacing": image_geo.get("spacing", ""),
            "label_spacing": label_geo.get("spacing", ""),
            "image_orientation": image_geo.get("orientation", ""),
            "label_orientation": label_geo.get("orientation", ""),
            "shape_match": bool(image_path and label_path and image_geo.get("shape") == label_geo.get("shape")),
            "spacing_match": bool(image_path and label_path and image_geo.get("spacing") == label_geo.get("spacing")),
            "orientation_match": bool(image_path and label_path and image_geo.get("orientation") == label_geo.get("orientation")),
            "label_histogram": hist,
        }
        row.update(parsed)
        rows.append(row)
    return rows


def add_session_orders(rows: list[dict[str, Any]]) -> None:
    by_animal: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_animal[(str(row["cohort"]), str(row["animal_id_strict"]))].append(row)

    def _pattern(items: list[dict[str, Any]]) -> str:
        time_counts = Counter(str(row.get("time_raw", "")) for row in items)
        pattern_parts: list[str] = []
        for time_raw in sorted(time_counts, key=lambda value: (TIME_ORDER.get(value, 1000 + SESSION_ORDER.get(value, 999)), value)):
            pattern_parts.extend([time_raw] * time_counts[time_raw])
        return " / ".join(pattern_parts)

    for items in by_animal.values():
        for idx, row in enumerate(sorted(items, key=session_sort_key), start=1):
            row["session_order"] = idx
        pattern_all_files = _pattern(items)
        labeled_items = [row for row in items if bool(row.get("has_label"))]
        pattern_labeled = _pattern(labeled_items) if labeled_items else pattern_all_files
        for row in items:
            row["animal_timepoint_pattern"] = pattern_labeled
            row["animal_timepoint_pattern_all_files"] = pattern_all_files
            row["animal_num_cases"] = len(items)


def animal_group_folds(rows: list[dict[str, Any]], num_folds: int = 5) -> list[dict[str, Any]]:
    animal_payload: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["cohort"] != "EpiBios":
            continue
        animal = str(row["animal_id_strict"])
        payload = animal_payload.setdefault(animal, {"animal_id_strict": animal, "num_cases": 0, "total_v0_positive_voxels": 0})
        payload["num_cases"] += 1
        try:
            payload["total_v0_positive_voxels"] += int(row.get("v0_positive_voxels") or 0)
        except ValueError:
            pass
    ordered = sorted(animal_payload.values(), key=lambda item: (-int(item["total_v0_positive_voxels"]), item["animal_id_strict"]))
    pattern = list(range(1, num_folds + 1)) + list(range(num_folds, 0, -1))
    out = []
    for idx, item in enumerate(ordered):
        fold = pattern[idx % len(pattern)]
        item = dict(item)
        item["animalwise_fold"] = fold
        out.append(item)
    return sorted(out, key=lambda item: (item["animalwise_fold"], item["animal_id_strict"]))


def aramra_split_proposal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    animals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["cohort"] == "ARAMRA002" and row.get("has_label") and row.get("has_image"):
            animals[str(row["animal_id_strict"])].append(row)
    payloads = []
    for animal, items in animals.items():
        times = [str(item["time_raw"]) for item in items]
        complete_pair = "D9" in times and "M5" in times
        has_repeat_d9 = times.count("D9") > 1
        payloads.append(
            {
                "animal_id_strict": animal,
                "num_cases": len(items),
                "timepoint_pattern": " / ".join(sorted(times, key=lambda value: (TIME_ORDER.get(value, 999), value))),
                "has_complete_d9_m5": complete_pair,
                "has_repeat_d9": has_repeat_d9,
                "sort_score": (0 if complete_pair else 1, 0 if has_repeat_d9 else 1, animal),
            }
        )
    ordered = sorted(payloads, key=lambda item: item["sort_score"])
    out = []
    for idx, item in enumerate(ordered):
        if idx < 24:
            split = "proposed_locked_external_test"
        elif idx < 36:
            split = "proposed_target_validation"
        else:
            split = "proposed_target_adapt_pool"
        out.append({key: value for key, value in item.items() if key != "sort_score"} | {"proposed_split": split})
    return out


def build_metadata(args: argparse.Namespace) -> None:
    paths = make_paths(Path(args.project_root), Path(args.out_dir), Path(args.aramra_root))
    for sub in ["metadata", "results", "logs", "status"]:
        ensure_dir(paths.out_dir / sub)
    workspace_cases, review_stats, rounds = read_workspace_db(paths)
    epi_rows = scan_epibios(paths, workspace_cases, review_stats)
    ar_rows = scan_aramra(paths)
    all_rows = epi_rows + ar_rows
    add_session_orders(all_rows)

    meta_dir = paths.out_dir / "metadata"
    fieldnames = [
        "case_id",
        "cohort",
        "animal_id_strict",
        "animal_id_raw",
        "animal_family",
        "scan_date",
        "time_raw",
        "time_numeric",
        "time_class",
        "session_ordinal",
        "session_order",
        "animal_timepoint_pattern",
        "animal_timepoint_pattern_all_files",
        "animal_num_cases",
        "field_strength",
        "field_strength_known",
        "original_fold",
        "revised_status",
        "label_round",
        "label_trust",
        "split_new",
        "has_image",
        "has_label",
        "image_path",
        "label_path",
        "image_shape",
        "label_shape",
        "image_spacing",
        "label_spacing",
        "image_orientation",
        "label_orientation",
        "shape_match",
        "spacing_match",
        "orientation_match",
        "v0_positive_voxels",
        "label_projection",
        "label_histogram",
    ]
    write_csv(meta_dir / "metadata_master.csv", all_rows, fieldnames)
    write_csv(meta_dir / "epibios_cases.csv", epi_rows, fieldnames)
    write_csv(meta_dir / "aramra_cases.csv", ar_rows, fieldnames)
    write_csv(meta_dir / "epibios_animalwise_group_folds.csv", animal_group_folds(all_rows))
    write_csv(meta_dir / "aramra_animal_split_proposal.csv", aramra_split_proposal(all_rows))

    summary_rows = summarize_metadata(all_rows)
    write_csv(meta_dir / "cohort_timepoint_summary.csv", summary_rows)
    field_rows = summarize_fields(all_rows)
    write_csv(meta_dir / "field_strength_summary.csv", field_rows)
    write_csv(meta_dir / "workspace_v0_training_summary.csv", workspace_training_summary(paths))
    integrity = data_integrity_summary(all_rows, rounds)
    write_json(meta_dir / "data_integrity_summary.json", integrity)
    print(json.dumps(integrity, ensure_ascii=False, indent=2, sort_keys=True))


def summarize_metadata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["cohort"]), str(row.get("time_raw", "")), str(row.get("animal_timepoint_pattern", "")))].append(row)
    out = []
    for (cohort, time_raw, pattern), items in sorted(groups.items()):
        out.append(
            {
                "cohort": cohort,
                "time_raw": time_raw,
                "animal_timepoint_pattern": pattern,
                "num_cases": len(items),
                "num_animals": len({item["animal_id_strict"] for item in items}),
                "num_labeled_cases": sum(1 for item in items if bool(item.get("has_label"))),
                "num_imaged_cases": sum(1 for item in items if bool(item.get("has_image"))),
            }
        )
    return out


def summarize_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["cohort"]), str(row.get("field_strength", "")), str(row.get("animal_family", "")))].append(row)
    out = []
    for (cohort, field, family), items in sorted(groups.items()):
        out.append(
            {
                "cohort": cohort,
                "field_strength": field,
                "animal_family_or_series": family,
                "num_cases": len(items),
                "num_animals": len({item["animal_id_strict"] for item in items}),
                "field_strength_known": all(bool(item.get("field_strength_known")) for item in items),
            }
        )
    return out


def workspace_training_summary(paths: Paths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_index in [0, 1]:
        report_dir = paths.workspace_v0 / "reports" / f"round_{round_index}"
        for fold_id in range(1, 6):
            train_csv = report_dir / f"fold_{fold_id}_train.csv"
            status_json = report_dir / f"fold_{fold_id}_train_status.json"
            csv_rows = read_csv(train_csv) if train_csv.exists() else []
            status = read_json(status_json) if status_json.exists() else {}
            rows.append(
                {
                    "round_index": round_index,
                    "fold_id": fold_id,
                    "train_csv": str(train_csv),
                    "status_json": str(status_json),
                    "num_logged_epochs": len(csv_rows),
                    "last_logged_epoch": max([int(row["epoch"]) for row in csv_rows], default=0),
                    "status_epochs_total": status.get("epochs_total", ""),
                    "status_epochs_completed": status.get("epochs_completed", ""),
                    "num_train_cases_after_val_split": status.get("num_train_cases", ""),
                    "num_val_cases": status.get("num_val_cases", ""),
                    "num_steps_per_epoch": status.get("num_steps_per_epoch", ""),
                    "best_epoch": status.get("best_epoch", ""),
                    "best_metric_name": status.get("best_metric_name", ""),
                    "best_metric_value": status.get("best_metric_value", ""),
                    "checkpoint_path": localized_workspace_path(paths, status.get("checkpoint_path")),
                    "last_checkpoint_path": localized_workspace_path(paths, status.get("last_checkpoint_path")),
                }
            )
    return rows


def data_integrity_summary(rows: list[dict[str, Any]], rounds: dict[str, Any]) -> dict[str, Any]:
    epi = [row for row in rows if row["cohort"] == "EpiBios"]
    ar = [row for row in rows if row["cohort"] == "ARAMRA002"]
    ar_labeled = [row for row in ar if row.get("has_label")]
    ar_eval = [row for row in ar if row.get("has_label") and row.get("has_image")]
    ar_unmatched_labels = [row["case_id"] for row in ar if row.get("has_label") and not row.get("has_image")]
    ar_unmatched_images = [row["case_id"] for row in ar if row.get("has_image") and not row.get("has_label")]
    return {
        "project_root": str(DEFAULT_PROJECT_ROOT),
        "num_total_rows_in_master": len(rows),
        "epibios": {
            "num_cases": len(epi),
            "num_animals": len({row["animal_id_strict"] for row in epi}),
            "num_reviewed_cases": sum(1 for row in epi if row.get("revised_status") in {"routine", "audit"}),
            "num_routine_cases": sum(1 for row in epi if row.get("revised_status") == "routine"),
            "num_audit_cases": sum(1 for row in epi if row.get("revised_status") == "audit"),
            "shape_mismatches": [row["case_id"] for row in epi if not bool(row.get("shape_match"))],
            "spacing_mismatches": [row["case_id"] for row in epi if not bool(row.get("spacing_match"))],
            "field_strength_status": "per-case field strength is not encoded in local filenames or workspace_v0 metadata; kept as unknown_epibios_mixed",
        },
        "aramra002": {
            "num_rows": len(ar),
            "num_labeled_cases": len(ar_labeled),
            "num_evaluable_labeled_cases": len(ar_eval),
            "num_animals_strict_labeled": len({row["animal_id_strict"] for row in ar_labeled}),
            "num_animals_strict_evaluable": len({row["animal_id_strict"] for row in ar_eval}),
            "unmatched_label_case_ids": ar_unmatched_labels,
            "unmatched_image_case_ids": ar_unmatched_images,
            "shape_mismatches": [row["case_id"] for row in ar_eval if not bool(row.get("shape_match"))],
            "spacing_mismatches": [row["case_id"] for row in ar_eval if not bool(row.get("spacing_match"))],
        },
        "workspace_v0_rounds": rounds,
    }


def load_models(checkpoint_paths: list[Path], device: torch.device) -> list[torch.nn.Module]:
    models = []
    for path in checkpoint_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = ResidualUNet3D(
            in_channels=1,
            out_channels=1,
            stage_channels=(32, 64, 96, 160, 256, 320),
            dropout_bottleneck=0.1,
        ).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        models.append(model)
    return models


def predict_ensemble(models: list[torch.nn.Module], image: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    total = len(models) * len(TTA_MODES)
    pred_sum = np.zeros(image.shape, dtype=np.float32)
    pred_sq_sum = np.zeros(image.shape, dtype=np.float32)
    for model in models:
        for mode in TTA_MODES:
            augmented = apply_tta(image, mode)
            pred = sliding_window_predict(model, augmented, device, PATCH_SIZE, OVERLAP)
            pred = invert_tta(pred, mode).astype(np.float32, copy=False)
            pred_sum += pred
            pred_sq_sum += pred * pred
    mean_pred = pred_sum / float(total)
    variance = np.maximum(pred_sq_sum / float(total) - mean_pred * mean_pred, 0.0)
    return mean_pred.astype(np.float32), variance.astype(np.float32)


def prediction_paths(out_dir: Path, round_index: int, case_id: str) -> dict[str, Path]:
    root = out_dir / "predictions" / "aramra" / f"round_{round_index}"
    return {
        "prob": root / "probabilities" / f"{case_id}.npz",
        "raw": root / "masks_raw" / f"{case_id}.nii.gz",
        "post_min2": root / "masks_post_min2" / f"{case_id}.nii.gz",
        "post_min16": root / "masks_post_min16" / f"{case_id}.nii.gz",
    }


def run_prediction(args: argparse.Namespace) -> None:
    paths = make_paths(Path(args.project_root), Path(args.out_dir), Path(args.aramra_root))
    metadata_path = paths.out_dir / "metadata" / "metadata_master.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run metadata first: {metadata_path}")
    rows = read_csv(metadata_path)
    ar_rows = [row for row in rows if row["cohort"] == "ARAMRA002" and row["has_label"] == "True" and row["has_image"] == "True"]
    if not ar_rows:
        raise RuntimeError("No evaluable ARAMRA rows found in metadata")
    if args.limit:
        ar_rows = ar_rows[: int(args.limit)]
    ensure_dir(paths.out_dir / "logs")
    ensure_dir(paths.out_dir / "status")
    log_path = paths.out_dir / "logs" / "predict_aramra.log"
    status_path = paths.out_dir / "status" / "predict_aramra_status.json"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    rounds = [int(item) for item in args.rounds.split(",")]
    start_all = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== prediction start {time.strftime('%Y-%m-%d %H:%M:%S')} rounds={rounds} cases={len(ar_rows)} device={device} ===\n")
        for round_index in rounds:
            checkpoint_dir = paths.workspace_v0 / "artifacts" / "checkpoints" / f"round_{round_index}"
            checkpoint_paths = [checkpoint_dir / f"fold_{fold}.pt" for fold in range(1, 6)]
            missing = [str(path) for path in checkpoint_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing checkpoint(s): {missing}")
            log.write(f"round {round_index}: loading models from {checkpoint_dir}\n")
            log.flush()
            models = load_models(checkpoint_paths, device)
            for idx, row in enumerate(ar_rows, start=1):
                case_id = row["case_id"]
                out_paths = prediction_paths(paths.out_dir, round_index, case_id)
                complete = all(path.exists() for path in out_paths.values())
                if complete and not args.overwrite:
                    message = f"round {round_index} case {idx}/{len(ar_rows)} {case_id}: skip existing"
                    log.write(message + "\n")
                    log.flush()
                    write_json(status_path, {"status": "running", "round_index": round_index, "case_index": idx, "num_cases": len(ar_rows), "case_id": case_id, "action": "skip_existing"})
                    continue
                image_path = Path(row["image_path"])
                image_data, affine, header = load_nifti(image_path, dtype=np.float32)
                image_pre = percentile_zscore(image_data.astype(np.float32), 0.5, 99.5)
                case_start = time.time()
                prob, var = predict_ensemble(models, image_pre, device)
                for path in out_paths.values():
                    ensure_dir(path.parent)
                np.savez_compressed(out_paths["prob"], probability=prob.astype(np.float16), uncertainty=var.astype(np.float16), case_id=case_id, round_index=round_index)
                save_nifti_like(out_paths["raw"], (prob >= THRESHOLD).astype(np.uint8), affine, header, np.uint8)
                save_nifti_like(out_paths["post_min2"], postprocess_probability_map(prob, THRESHOLD, min_component_voxels=2, largest_only=False), affine, header, np.uint8)
                save_nifti_like(out_paths["post_min16"], postprocess_probability_map(prob, THRESHOLD, min_component_voxels=16, largest_only=False), affine, header, np.uint8)
                elapsed = time.time() - case_start
                message = f"round {round_index} case {idx}/{len(ar_rows)} {case_id}: completed in {elapsed:.1f}s"
                log.write(message + "\n")
                log.flush()
                write_json(status_path, {"status": "running", "round_index": round_index, "case_index": idx, "num_cases": len(ar_rows), "case_id": case_id, "elapsed_seconds": elapsed})
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_json(status_path, {"status": "completed", "rounds": rounds, "num_cases": len(ar_rows), "elapsed_seconds": time.time() - start_all})
        log.write(f"=== prediction complete elapsed={time.time() - start_all:.1f}s ===\n")


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def hd95(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, float, float]) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("nan")
    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = np.logical_xor(pred, ndimage.binary_erosion(pred, structure=structure, border_value=0))
    target_surface = np.logical_xor(target, ndimage.binary_erosion(target, structure=structure, border_value=0))
    if not pred_surface.any() or not target_surface.any():
        return float("nan")
    dist_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    dist_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dist_to_target[pred_surface], dist_to_pred[target_surface]])
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def component_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    pred_lab, pred_n = ndimage.label(pred.astype(np.uint8), structure=structure)
    target_lab, target_n = ndimage.label(target.astype(np.uint8), structure=structure)
    overlap = pred.astype(bool) & target.astype(bool)
    hit_pred = set(int(v) for v in np.unique(pred_lab[overlap]) if int(v) != 0)
    hit_target = set(int(v) for v in np.unique(target_lab[overlap]) if int(v) != 0)
    tp_pred = len(hit_pred)
    tp_target = len(hit_target)
    precision = tp_pred / pred_n if pred_n else (1.0 if target_n == 0 else 0.0)
    recall = tp_target / target_n if target_n else (1.0 if pred_n == 0 else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "pred_components": int(pred_n),
        "gt_components": int(target_n),
        "lesion_precision": float(precision),
        "lesion_recall": float(recall),
        "lesion_f1": float(f1),
        "pred_components_hit": int(tp_pred),
        "gt_components_hit": int(tp_target),
    }


def spacing_from_header(path: Path) -> tuple[float, float, float]:
    img = nib.load(str(path))
    return tuple(float(v) for v in img.header.get_zooms()[:3])


def metric_row(case_row: dict[str, str], pred_mask: np.ndarray, target: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, Any]:
    inter = int(np.logical_and(pred_mask.astype(bool), target.astype(bool)).sum())
    pred_vox = int(pred_mask.sum())
    gt_vox = int(target.sum())
    comp = component_metrics(pred_mask, target)
    return {
        "case_id": case_row["case_id"],
        "cohort": case_row["cohort"],
        "animal_id_strict": case_row["animal_id_strict"],
        "animal_id_raw": case_row["animal_id_raw"],
        "animal_family": case_row["animal_family"],
        "time_raw": case_row["time_raw"],
        "time_class": case_row["time_class"],
        "session_order": case_row.get("session_order", ""),
        "animal_timepoint_pattern": case_row.get("animal_timepoint_pattern", ""),
        "original_fold": case_row.get("original_fold", ""),
        "revised_status": case_row.get("revised_status", ""),
        "label_trust": case_row.get("label_trust", ""),
        "field_strength": case_row.get("field_strength", ""),
        "dice": dice(pred_mask, target),
        "hd95": hd95(pred_mask, target, spacing),
        "intersection": inter,
        "pred_positive_voxels": pred_vox,
        "gt_positive_voxels": gt_vox,
        "fp_voxels": int(np.logical_and(pred_mask.astype(bool), ~target.astype(bool)).sum()),
        "fn_voxels": int(np.logical_and(~pred_mask.astype(bool), target.astype(bool)).sum()),
        "absolute_volume_error_voxels": abs(pred_vox - gt_vox),
        "signed_volume_error_voxels": pred_vox - gt_vox,
        "relative_volume_error": (pred_vox - gt_vox) / gt_vox if gt_vox else float("nan"),
        **comp,
    }


def micro_dice(rows: list[dict[str, Any]]) -> float:
    inter = sum(float(row["intersection"]) for row in rows)
    pred = sum(float(row["pred_positive_voxels"]) for row in rows)
    gt = sum(float(row["gt_positive_voxels"]) for row in rows)
    denom = pred + gt
    return 1.0 if denom == 0 else float(2.0 * inter / denom)


def nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def nanmedian(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def aggregate_metrics(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    out = []
    for key_values, items in sorted(groups.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        item: dict[str, Any] = {key: value for key, value in zip(keys, key_values)}
        animal_means = []
        by_animal: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in items:
            by_animal[str(row["animal_id_strict"])].append(row)
        for animal_items in by_animal.values():
            animal_means.append(float(np.mean([float(row["dice"]) for row in animal_items])))
        item.update(
            {
                "num_cases": len(items),
                "num_animals": len(by_animal),
                "macro_dice": float(np.mean([float(row["dice"]) for row in items])) if items else float("nan"),
                "animal_macro_dice": float(np.mean(animal_means)) if animal_means else float("nan"),
                "micro_dice": micro_dice(items),
                "median_hd95": nanmedian([float(row["hd95"]) for row in items]),
                "mean_hd95": nanmean([float(row["hd95"]) for row in items]),
                "mean_lesion_f1": nanmean([float(row["lesion_f1"]) for row in items]),
                "mean_absolute_volume_error_voxels": nanmean([float(row["absolute_volume_error_voxels"]) for row in items]),
            }
        )
        out.append(item)
    return out


def animal_bootstrap(rows: list[dict[str, Any]], keys: list[str], n_boot: int = 2000) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    out = []
    for key_values, items in sorted(groups.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        by_animal: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in items:
            by_animal[str(row["animal_id_strict"])].append(row)
        animals = sorted(by_animal)
        if not animals:
            continue
        animal_values = np.asarray([np.mean([float(row["dice"]) for row in by_animal[animal]]) for animal in animals], dtype=np.float64)
        if len(animals) == 1:
            low = high = mean = float(animal_values[0])
        else:
            samples = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(animals), size=len(animals))
                samples.append(float(animal_values[idx].mean()))
            low, high = np.percentile(samples, [2.5, 97.5])
            mean = float(animal_values.mean())
        item = {key: value for key, value in zip(keys, key_values)}
        item.update({"animal_macro_dice_mean": mean, "animal_macro_dice_ci95_low": float(low), "animal_macro_dice_ci95_high": float(high), "num_animals": len(animals)})
        out.append(item)
    return out


def compute_leakage(epi_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_animal: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in epi_rows:
        by_animal[row["animal_id_strict"]].append(row)
    case_rows = []
    for row in epi_rows:
        fold = str(row["original_fold"])
        siblings = [sib for sib in by_animal[row["animal_id_strict"]] if sib["case_id"] != row["case_id"]]
        train_siblings = [sib for sib in siblings if str(sib["original_fold"]) != fold]
        case_rows.append(
            {
                "case_id": row["case_id"],
                "animal_id_strict": row["animal_id_strict"],
                "time_raw": row["time_raw"],
                "animal_family": row["animal_family"],
                "fold_id": fold,
                "num_same_animal_cases_total": len(by_animal[row["animal_id_strict"]]),
                "num_same_animal_train_siblings": len(train_siblings),
                "has_same_animal_train_sibling": len(train_siblings) > 0,
                "train_sibling_timepoints": " / ".join(sorted(sib["time_raw"] for sib in train_siblings)),
                "train_sibling_case_ids": ";".join(sorted(sib["case_id"] for sib in train_siblings)),
            }
        )
    fold_rows = []
    for fold in sorted({str(row["original_fold"]) for row in epi_rows}):
        items = [row for row in case_rows if row["fold_id"] == fold]
        holdout_animals = {row["animal_id_strict"] for row in items}
        leaked_animals = {row["animal_id_strict"] for row in items if row["has_same_animal_train_sibling"]}
        fold_rows.append(
            {
                "fold_id": fold,
                "holdout_cases": len(items),
                "holdout_animals": len(holdout_animals),
                "holdout_cases_with_train_sibling": sum(1 for row in items if row["has_same_animal_train_sibling"]),
                "holdout_animals_with_train_sibling": len(leaked_animals),
                "animal_overlap_rate_cases": sum(1 for row in items if row["has_same_animal_train_sibling"]) / len(items) if items else float("nan"),
                "animal_overlap_rate_animals": len(leaked_animals) / len(holdout_animals) if holdout_animals else float("nan"),
            }
        )
    return case_rows, fold_rows


def load_oof_probability(paths: Paths, round_index: int, case_id: str) -> np.ndarray:
    payload = np.load(paths.workspace_v0 / "artifacts" / "oof" / f"round_{round_index}" / f"{case_id}.npz")
    return np.asarray(payload["s"], dtype=np.float32)


def load_workspace_label(paths: Paths, round_index: int, case_id: str) -> np.ndarray:
    return load_label_bool(paths.workspace_v0 / "artifacts" / "labels" / "binary" / f"round_{round_index}" / f"{case_id}.nii.gz")


def analyze_static_reference(paths: Paths, epi_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for pred_round in [0, 1]:
        for ref_round in [0, 1]:
            for case in epi_rows:
                case_id = case["case_id"]
                prob = load_oof_probability(paths, pred_round, case_id)
                target = load_workspace_label(paths, ref_round, case_id)
                raw = (prob >= THRESHOLD).astype(np.uint8)
                post2 = postprocess_probability_map(prob, THRESHOLD, min_component_voxels=2, largest_only=False)
                post16 = postprocess_probability_map(prob, THRESHOLD, min_component_voxels=16, largest_only=False)
                spacing = spacing_from_header(paths.workspace_v0 / "artifacts" / "labels" / "binary" / f"round_{ref_round}" / f"{case_id}.nii.gz")
                for variant, pred_mask in [("raw", raw), ("post_min2", post2), ("post_min16", post16)]:
                    metric = metric_row(case, pred_mask, target, spacing)
                    metric["prediction_round"] = pred_round
                    metric["reference_round"] = ref_round
                    metric["postprocess_variant"] = variant
                    rows.append(metric)
    summary = aggregate_metrics(rows, ["prediction_round", "reference_round", "postprocess_variant"])
    summary += aggregate_metrics(rows, ["prediction_round", "reference_round", "postprocess_variant", "revised_status"])
    summary += aggregate_metrics(rows, ["prediction_round", "reference_round", "postprocess_variant", "time_raw"])
    summary += aggregate_metrics(rows, ["prediction_round", "reference_round", "postprocess_variant", "animal_family"])
    return rows, summary


def analyze_aramra_external(paths: Paths, ar_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    missing = []
    for round_index in [0, 1]:
        for case in ar_rows:
            case_id = case["case_id"]
            pmap = prediction_paths(paths.out_dir, round_index, case_id)
            if not pmap["prob"].exists():
                missing.append(str(pmap["prob"]))
                continue
            target = load_label_bool(Path(case["label_path"]))
            spacing = spacing_from_header(Path(case["label_path"]))
            with np.load(pmap["prob"]) as payload:
                prob = np.asarray(payload["probability"], dtype=np.float32)
            masks = {
                "raw": (prob >= THRESHOLD).astype(np.uint8),
                "post_min2": load_label_bool(pmap["post_min2"]) if pmap["post_min2"].exists() else postprocess_probability_map(prob, THRESHOLD, 2, False),
                "post_min16": load_label_bool(pmap["post_min16"]) if pmap["post_min16"].exists() else postprocess_probability_map(prob, THRESHOLD, 16, False),
            }
            for variant, mask in masks.items():
                metric = metric_row(case, mask, target, spacing)
                metric["prediction_round"] = round_index
                metric["postprocess_variant"] = variant
                rows.append(metric)
    if missing:
        raise FileNotFoundError(f"Missing ARAMRA prediction files; first five: {missing[:5]}")
    group_keys = [
        ["prediction_round", "postprocess_variant"],
        ["prediction_round", "postprocess_variant", "time_raw"],
        ["prediction_round", "postprocess_variant", "animal_timepoint_pattern"],
        ["prediction_round", "postprocess_variant", "animal_family"],
    ]
    group_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    for keys in group_keys:
        group_rows.extend(aggregate_metrics(rows, keys))
        ci_rows.extend(animal_bootstrap(rows, keys))
    animal_rows = aggregate_metrics(rows, ["prediction_round", "postprocess_variant", "animal_id_strict"])
    pair_rows = paired_longitudinal_rows(rows)
    repeat_rows = repeat_d9_rows(paths, ar_rows)
    return rows, group_rows, animal_rows, pair_rows + repeat_rows + ci_rows


def paired_longitudinal_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        groups[(int(row["prediction_round"]), str(row["postprocess_variant"]), str(row["animal_id_strict"]))].append(row)
    out = []
    for (round_index, variant, animal), items in sorted(groups.items()):
        d9 = [row for row in items if row["time_raw"] == "D9"]
        m5 = [row for row in items if row["time_raw"] == "M5"]
        if not d9 or not m5:
            continue
        for d9_row in d9:
            for m5_row in m5:
                out.append(
                    {
                        "analysis_type": "d9_m5_pair_metric_delta",
                        "prediction_round": round_index,
                        "postprocess_variant": variant,
                        "animal_id_strict": animal,
                        "d9_case_id": d9_row["case_id"],
                        "m5_case_id": m5_row["case_id"],
                        "d9_dice": d9_row["dice"],
                        "m5_dice": m5_row["dice"],
                        "dice_delta_m5_minus_d9": float(m5_row["dice"]) - float(d9_row["dice"]),
                        "gt_volume_delta_m5_minus_d9": int(m5_row["gt_positive_voxels"]) - int(d9_row["gt_positive_voxels"]),
                        "pred_volume_delta_m5_minus_d9": int(m5_row["pred_positive_voxels"]) - int(d9_row["pred_positive_voxels"]),
                        "num_d9_for_animal": len(d9),
                        "num_m5_for_animal": len(m5),
                    }
                )
    return out


def repeat_d9_rows(paths: Paths, ar_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    by_animal: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ar_rows:
        if row["time_raw"] == "D9":
            by_animal[row["animal_id_strict"]].append(row)
    for animal, items in sorted(by_animal.items()):
        if len(items) < 2:
            continue
        for round_index in [0, 1]:
            for variant in ["raw", "post_min2", "post_min16"]:
                for a, b in combinations(sorted(items, key=lambda row: row["case_id"]), 2):
                    path_a = prediction_paths(paths.out_dir, round_index, a["case_id"])
                    path_b = prediction_paths(paths.out_dir, round_index, b["case_id"])
                    if variant == "raw":
                        with np.load(path_a["prob"]) as za:
                            pa = np.asarray(za["probability"], dtype=np.float32) >= THRESHOLD
                        with np.load(path_b["prob"]) as zb:
                            pb = np.asarray(zb["probability"], dtype=np.float32) >= THRESHOLD
                    else:
                        pa = load_label_bool(path_a[variant])
                        pb = load_label_bool(path_b[variant])
                    la = load_label_bool(Path(a["label_path"]))
                    lb = load_label_bool(Path(b["label_path"]))
                    rows.append(
                        {
                            "analysis_type": "repeat_d9_scan_rescan_consistency",
                            "prediction_round": round_index,
                            "postprocess_variant": variant,
                            "animal_id_strict": animal,
                            "case_id_a": a["case_id"],
                            "case_id_b": b["case_id"],
                            "prediction_mask_dice_between_repeats": dice(pa.astype(np.uint8), pb.astype(np.uint8)),
                            "label_mask_dice_between_repeats": dice(la, lb),
                            "num_d9_for_animal": len(items),
                        }
                    )
    return rows


def run_analysis(args: argparse.Namespace) -> None:
    paths = make_paths(Path(args.project_root), Path(args.out_dir), Path(args.aramra_root))
    meta = read_csv(paths.out_dir / "metadata" / "metadata_master.csv")
    epi_rows = [row for row in meta if row["cohort"] == "EpiBios"]
    ar_rows = [row for row in meta if row["cohort"] == "ARAMRA002" and row["has_label"] == "True" and row["has_image"] == "True"]
    result_dir = ensure_dir(paths.out_dir / "results")

    leak_case, leak_fold = compute_leakage(epi_rows)
    write_csv(result_dir / "original_fold_animal_leakage_cases.csv", leak_case)
    write_csv(result_dir / "original_fold_animal_leakage_by_fold.csv", leak_fold)

    static_rows, static_summary = analyze_static_reference(paths, epi_rows)
    write_csv(result_dir / "static_reference_case_metrics.csv", static_rows)
    write_csv(result_dir / "static_reference_summary.csv", static_summary)

    external_rows, external_summary, animal_summary, paired_rows = analyze_aramra_external(paths, ar_rows)
    write_csv(result_dir / "aramra_external_case_metrics.csv", external_rows)
    write_csv(result_dir / "aramra_external_group_metrics.csv", external_summary)
    write_csv(result_dir / "aramra_external_animal_metrics.csv", animal_summary)
    write_csv(result_dir / "aramra_longitudinal_pair_and_bootstrap_metrics.csv", paired_rows)

    review_rows = review_distribution(epi_rows)
    write_csv(result_dir / "review_distribution.csv", review_rows)

    summary = build_result_summary(paths, leak_case, leak_fold, static_summary, external_summary, paired_rows)
    write_json(result_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def review_distribution(epi_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for keys in [["revised_status"], ["revised_status", "animal_family"], ["revised_status", "time_raw"], ["revised_status", "original_fold"]]:
        groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in epi_rows:
            groups[tuple(row.get(key, "") for key in keys)].append(row)
        for key_values, items in sorted(groups.items()):
            payload = {key: value for key, value in zip(keys, key_values)}
            payload.update({"num_cases": len(items), "num_animals": len({row["animal_id_strict"] for row in items})})
            out.append(payload)
    return out


def build_result_summary(
    paths: Paths,
    leak_case: list[dict[str, Any]],
    leak_fold: list[dict[str, Any]],
    static_summary: list[dict[str, Any]],
    external_summary: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    static_overall = [
        row
        for row in static_summary
        if set(row.keys()).issuperset({"prediction_round", "reference_round", "postprocess_variant", "num_cases"})
        and "revised_status" not in row
        and "time_raw" not in row
        and "animal_family" not in row
    ]
    external_overall = [
        row
        for row in external_summary
        if set(row.keys()).issuperset({"prediction_round", "postprocess_variant", "num_cases"})
        and "time_raw" not in row
        and "animal_timepoint_pattern" not in row
        and "animal_family" not in row
    ]
    pair = [row for row in paired_rows if row.get("analysis_type") == "d9_m5_pair_metric_delta"]
    repeat = [row for row in paired_rows if row.get("analysis_type") == "repeat_d9_scan_rescan_consistency"]
    return {
        "output_dir": str(paths.out_dir),
        "leakage": {
            "num_cases": len(leak_case),
            "cases_with_same_animal_train_sibling": sum(1 for row in leak_case if bool(row["has_same_animal_train_sibling"])),
            "case_overlap_rate": sum(1 for row in leak_case if bool(row["has_same_animal_train_sibling"])) / len(leak_case) if leak_case else float("nan"),
            "fold_rows": leak_fold,
        },
        "static_reference_overall": static_overall,
        "aramra_external_overall": external_overall,
        "num_d9_m5_pair_rows": len(pair),
        "num_repeat_d9_rows": len(repeat),
    }


def md_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                if math.isnan(value):
                    cells.append("")
                else:
                    cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body)


def run_report(args: argparse.Namespace) -> None:
    paths = make_paths(Path(args.project_root), Path(args.out_dir), Path(args.aramra_root))
    meta_summary = read_json(paths.out_dir / "metadata" / "data_integrity_summary.json")
    result_summary = read_json(paths.out_dir / "results" / "summary.json")
    static_rows = read_csv(paths.out_dir / "results" / "static_reference_summary.csv")
    external_rows = read_csv(paths.out_dir / "results" / "aramra_external_group_metrics.csv")
    leakage_fold = read_csv(paths.out_dir / "results" / "original_fold_animal_leakage_by_fold.csv")
    review_dist = read_csv(paths.out_dir / "results" / "review_distribution.csv")
    training_rows = read_csv(paths.out_dir / "metadata" / "workspace_v0_training_summary.csv")

    def convert_numeric(row: dict[str, str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                out[key] = value
                continue
            try:
                out[key] = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                out[key] = value
        return out

    static_num = [convert_numeric(row) for row in static_rows]
    external_num = [convert_numeric(row) for row in external_rows]
    leakage_num = [convert_numeric(row) for row in leakage_fold]
    review_num = [convert_numeric(row) for row in review_dist]
    training_num = [convert_numeric(row) for row in training_rows]
    static_overall = [
        row
        for row in static_num
        if {"prediction_round", "reference_round", "postprocess_variant"}.issubset(row)
        and not row.get("revised_status")
        and not row.get("time_raw")
        and not row.get("animal_family")
    ]
    external_overall = [
        row
        for row in external_num
        if {"prediction_round", "postprocess_variant"}.issubset(row)
        and not row.get("time_raw")
        and not row.get("animal_timepoint_pattern")
        and not row.get("animal_family")
    ]
    external_time = [row for row in external_num if "time_raw" in row and row.get("time_raw") in {"D9", "M5"}]
    report = []
    report.append("# workspace_v0 full external analysis report")
    report.append("")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Scope")
    report.append("")
    report.append("This analysis uses the archived `workspace_v0` round0 and round1 5-fold models only. It does not use later correction attempts or the current SGRA revision-policy branch for prediction.")
    report.append("")
    report.append("## workspace_v0 training audit")
    report.append("")
    report.append(md_table(training_num, ["round_index", "fold_id", "status_epochs_total", "last_logged_epoch", "best_epoch", "best_metric_value", "num_train_cases_after_val_split", "num_val_cases"]))
    report.append("")
    report.append("The current local `configs/model.yaml` has later SGRA-related changes and `min_component_voxels=2`; the original GitHub config used `min_component_voxels=16`. To avoid config drift, external metrics are reported for raw, `post_min2`, and `post_min16` masks.")
    report.append("")
    report.append("## Data inventory")
    report.append("")
    report.append(f"- EpiBios: {meta_summary['epibios']['num_cases']} cases, {meta_summary['epibios']['num_animals']} animals, {meta_summary['epibios']['num_reviewed_cases']} reviewed cases ({meta_summary['epibios']['num_routine_cases']} routine, {meta_summary['epibios']['num_audit_cases']} audit).")
    report.append(f"- ARAMRA002: {meta_summary['aramra002']['num_labeled_cases']} labeled cases, {meta_summary['aramra002']['num_evaluable_labeled_cases']} evaluable image-label cases, {meta_summary['aramra002']['num_animals_strict_evaluable']} strict animals.")
    report.append(f"- ARAMRA unmatched labeled cases: {len(meta_summary['aramra002']['unmatched_label_case_ids'])}. Unmatched images: {len(meta_summary['aramra002']['unmatched_image_case_ids'])}.")
    report.append(f"- EpiBios field strength: {meta_summary['epibios']['field_strength_status']}.")
    report.append("")
    report.append("## Original fold animal leakage")
    report.append("")
    report.append(md_table(leakage_num, ["fold_id", "holdout_cases", "holdout_animals", "holdout_cases_with_train_sibling", "holdout_animals_with_train_sibling", "animal_overlap_rate_cases", "animal_overlap_rate_animals"]))
    report.append("")
    leak = result_summary["leakage"]
    report.append(f"Overall, {leak['cases_with_same_animal_train_sibling']}/{leak['num_cases']} holdout cases have same-animal training siblings; case overlap rate = {leak['case_overlap_rate']:.6f}.")
    report.append("")
    report.append("## Static-reference matrix")
    report.append("")
    report.append(md_table(static_overall, ["prediction_round", "reference_round", "postprocess_variant", "num_cases", "macro_dice", "micro_dice", "animal_macro_dice", "median_hd95"]))
    report.append("")
    report.append("Interpretation: compare round0 prediction vs round1 reference against round1 prediction vs round1 reference to separate label-reference shift from finetune model gain.")
    report.append("")
    report.append("## ARAMRA002 external evaluation")
    report.append("")
    report.append(md_table(external_overall, ["prediction_round", "postprocess_variant", "num_cases", "num_animals", "macro_dice", "animal_macro_dice", "micro_dice", "median_hd95", "mean_lesion_f1"]))
    report.append("")
    report.append("### Timepoint split")
    report.append("")
    report.append(md_table(external_time, ["prediction_round", "postprocess_variant", "time_raw", "num_cases", "num_animals", "macro_dice", "animal_macro_dice", "micro_dice", "median_hd95", "mean_lesion_f1"]))
    report.append("")
    report.append("## Review distribution")
    report.append("")
    report.append(md_table(review_num, [col for col in ["revised_status", "animal_family", "time_raw", "original_fold", "num_cases", "num_animals"] if any(col in row for row in review_num)], max_rows=80))
    report.append("")
    report.append("## Output files")
    report.append("")
    report.append("- `metadata/metadata_master.csv`: unified EpiBios + ARAMRA case table.")
    report.append("- `metadata/workspace_v0_training_summary.csv`: exact logged epoch counts and selected validation checkpoints for archived round0/round1 folds.")
    report.append("- `results/original_fold_animal_leakage_cases.csv` and `results/original_fold_animal_leakage_by_fold.csv`: leakage audit.")
    report.append("- `results/static_reference_case_metrics.csv` and `results/static_reference_summary.csv`: 2x2 prediction/reference matrix.")
    report.append("- `predictions/aramra/round_*/`: external probabilities and masks.")
    report.append("- `results/aramra_external_case_metrics.csv`, `results/aramra_external_group_metrics.csv`, `results/aramra_external_animal_metrics.csv`: external metrics.")
    report.append("- `results/aramra_longitudinal_pair_and_bootstrap_metrics.csv`: D9/M5 pair deltas, repeat-D9 consistency, and animal bootstrap intervals.")
    report.append("")
    report.append("## Caveats")
    report.append("")
    report.append("- ARAMRA002 is an independent 9.4T target cohort, not a low-field external cohort.")
    report.append("- EpiBios per-case field strength is not recoverable from the local filenames or workspace metadata, so field-strength claims require an external protocol table.")
    report.append("- This report runs existing workspace_v0 model prediction and analysis. Animal-wise retraining is a separate experiment and should use the generated `metadata/epibios_animalwise_group_folds.csv` split table.")
    report_path = paths.out_dir / "report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="workspace_v0 full external analysis")
    parser.add_argument("command", choices=["metadata", "predict", "analyze", "report", "all"])
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_PROJECT_ROOT / "analysis" / "workspace_v0_full_external_analysis"))
    parser.add_argument("--aramra-root", default=r"E:\Hemorrhage")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", default="0,1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"metadata", "all"}:
        build_metadata(args)
    if args.command in {"predict", "all"}:
        run_prediction(args)
    if args.command in {"analyze", "all"}:
        run_analysis(args)
    if args.command in {"report", "all"}:
        run_report(args)


if __name__ == "__main__":
    main()
