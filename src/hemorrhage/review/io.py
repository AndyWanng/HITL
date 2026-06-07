"""Review bundle export, validation, and metadata normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hemorrhage.data.nifti import copy_nifti, label_histogram, load_nifti, project_binary_mask, read_metadata_csv, validate_label_codes, write_metadata_csv
from hemorrhage.protocol.metrics import dice_score, edit_ratio
from hemorrhage.utils import ensure_dir


def export_review_bundle(bundle_root: Path, rows: list[dict[str, str]]) -> Path:
    ensure_dir(bundle_root)
    ensure_dir(bundle_root / "images")
    ensure_dir(bundle_root / "labels_seed")
    if any(row.get("model_mask_path") for row in rows):
        ensure_dir(bundle_root / "model_mask")
    if any(row.get("uncertainty_path") for row in rows):
        ensure_dir(bundle_root / "uncertainty")
    write_metadata_csv(bundle_root / "manifest.csv", rows)
    return bundle_root


def normalize_review_metadata(input_csv: Path, output_csv: Path, required_fields: list[str]) -> Path:
    rows = read_metadata_csv(input_csv)
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized_row = {key.strip().lower(): str(value).strip() for key, value in row.items()}
        missing = [field for field in required_fields if not normalized_row.get(field)]
        if missing:
            raise ValueError(f"Missing required metadata fields {missing} in row {row}")
        normalized.append(normalized_row)
    write_metadata_csv(output_csv, normalized)
    return output_csv


def validate_import_dir(labels_dir: Path, metadata_csv: Path, required_fields: list[str]) -> dict[str, dict[str, str]]:
    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)
    if not metadata_csv.exists():
        raise FileNotFoundError(metadata_csv)
    metadata_rows = {row["case_id"]: row for row in read_metadata_csv(metadata_csv)}
    for case_id, row in metadata_rows.items():
        missing = [field for field in required_fields if field not in row or row[field] == ""]
        if missing:
            raise ValueError(f"Case {case_id} missing metadata fields {missing}")
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {case_id}: {label_path}")
    return metadata_rows


def import_review_label(label_path: Path) -> dict[str, Any]:
    raw = load_nifti(label_path)
    raw_codes = raw.data.astype(np.int16)
    validate_label_codes(raw_codes)
    binary = project_binary_mask(raw_codes)
    return {
        "raw": raw_codes,
        "binary": binary,
        "affine": raw.affine,
        "header": raw.header,
        "histogram": label_histogram(raw_codes),
    }


def compute_review_stats(previous_binary: np.ndarray, imported_binary: np.ndarray, anchor_binary: np.ndarray | None = None) -> dict[str, float]:
    modified_slices = int(np.any(previous_binary != imported_binary, axis=(0, 1)).sum())
    whole_volume_edit = float(np.mean(previous_binary != imported_binary))
    payload = {
        "modified_slices_count": float(modified_slices),
        "edit_ratio": edit_ratio(imported_binary, previous_binary),
        "whole_volume_edit_ratio": whole_volume_edit,
    }
    if anchor_binary is not None:
        payload["anchor_assisted_dice"] = float(dice_score(anchor_binary, imported_binary))
    return payload
