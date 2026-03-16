"""NIfTI IO, preprocessing, and label projection helpers."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

from hemorrhage.utils import ensure_dir

_SUBJECT_PATTERN = re.compile(r"(Rat\d+|MHR_\d+)")


@dataclass(slots=True)
class NiftiVolume:
    data: np.ndarray
    affine: np.ndarray
    header: nib.Nifti1Header


@dataclass(slots=True)
class CasePaths:
    case_id: str
    image_path: Path
    label_path: Path
    subject_id: str


@dataclass(slots=True)
class CaseGeometry:
    case_id: str
    image_shape: tuple[int, int, int]
    label_shape: tuple[int, int, int]
    image_spacing: tuple[float, float, float]
    label_spacing: tuple[float, float, float]
    image_orientation: tuple[str, str, str]
    label_orientation: tuple[str, str, str]
    shape_match: bool
    spacing_match: bool
    affine_match: bool


def load_nifti(path: Path) -> NiftiVolume:
    image = nib.load(str(path))
    return NiftiVolume(
        data=np.asarray(image.get_fdata(dtype=np.float32)),
        affine=image.affine.copy(),
        header=image.header.copy(),
    )


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, dtype: np.dtype) -> None:
    ensure_dir(path.parent)
    out = nib.Nifti1Image(data.astype(dtype), affine=affine, header=header.copy())
    out.set_data_dtype(dtype)
    nib.save(out, str(path))


def copy_nifti(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def project_binary_mask(raw_codes: np.ndarray) -> np.ndarray:
    return np.isin(raw_codes, (1, 3)).astype(np.uint8)


def validate_label_codes(raw_codes: np.ndarray) -> None:
    values = set(np.unique(raw_codes).tolist())
    if not values.issubset({0, 1, 2, 3}):
        raise ValueError(f"Unexpected label codes: {sorted(values)}")


def extract_case_id_from_image(image_path: Path) -> str:
    name = image_path.name
    if not name.endswith("_0000.nii.gz"):
        raise ValueError(f"Image name must end with _0000.nii.gz: {image_path}")
    return name.removesuffix("_0000.nii.gz")


def extract_subject_id(case_id: str) -> str:
    match = _SUBJECT_PATTERN.search(case_id)
    return match.group(1) if match else case_id


def scan_case_paths(data_root: Path) -> list[CasePaths]:
    images_dir = data_root / "imagesTr"
    labels_dir = data_root / "labelsTr"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("Expected data/imagesTr and data/labelsTr")

    cases: list[CasePaths] = []
    for image_path in sorted(images_dir.glob("*.nii.gz")):
        case_id = extract_case_id_from_image(image_path)
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {case_id}: {label_path}")
        cases.append(
            CasePaths(
                case_id=case_id,
                image_path=image_path.resolve(),
                label_path=label_path.resolve(),
                subject_id=extract_subject_id(case_id),
            )
        )
    return cases


def inspect_case_geometry(
    case_id: str,
    image_path: Path,
    label_path: Path,
    spacing_tolerance: float,
    affine_tolerance: float,
) -> CaseGeometry:
    image = nib.load(str(image_path))
    label = nib.load(str(label_path))
    image_shape = tuple(int(v) for v in image.shape[:3])
    label_shape = tuple(int(v) for v in label.shape[:3])
    image_spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    label_spacing = tuple(float(v) for v in label.header.get_zooms()[:3])
    image_orientation = tuple(str(v) for v in nib.aff2axcodes(image.affine))
    label_orientation = tuple(str(v) for v in nib.aff2axcodes(label.affine))
    return CaseGeometry(
        case_id=case_id,
        image_shape=image_shape,
        label_shape=label_shape,
        image_spacing=image_spacing,
        label_spacing=label_spacing,
        image_orientation=image_orientation,
        label_orientation=label_orientation,
        shape_match=image_shape == label_shape,
        spacing_match=bool(np.allclose(np.asarray(image_spacing), np.asarray(label_spacing), atol=spacing_tolerance, rtol=0.0)),
        affine_match=bool(np.allclose(image.affine, label.affine, atol=affine_tolerance, rtol=0.0)),
    )


def validate_case_geometry(case_geometry: CaseGeometry) -> None:
    if not case_geometry.shape_match:
        raise ValueError(
            f"Image/label shape mismatch for {case_geometry.case_id}: "
            f"{case_geometry.image_shape} vs {case_geometry.label_shape}"
        )
    if not case_geometry.spacing_match:
        raise ValueError(
            f"Image/label spacing mismatch for {case_geometry.case_id}: "
            f"{case_geometry.image_spacing} vs {case_geometry.label_spacing}"
        )
    if not case_geometry.affine_match:
        raise ValueError(f"Image/label affine mismatch for {case_geometry.case_id}")


def summarize_case_geometries(cases: Iterable[CaseGeometry]) -> dict[str, object]:
    cases = list(cases)
    return {
        "num_cases": len(cases),
        "all_shape_match": all(case.shape_match for case in cases),
        "all_spacing_match": all(case.spacing_match for case in cases),
        "all_affine_match": all(case.affine_match for case in cases),
        "unique_image_shapes": [list(item) for item in sorted({tuple(case.image_shape) for case in cases})],
        "unique_label_shapes": [list(item) for item in sorted({tuple(case.label_shape) for case in cases})],
        "unique_image_spacings": [list(item) for item in sorted({tuple(round(v, 6) for v in case.image_spacing) for case in cases})],
        "unique_label_spacings": [list(item) for item in sorted({tuple(round(v, 6) for v in case.label_spacing) for case in cases})],
        "cases": [
            {
                "case_id": case.case_id,
                "image_shape": list(case.image_shape),
                "label_shape": list(case.label_shape),
                "image_spacing": [round(v, 6) for v in case.image_spacing],
                "label_spacing": [round(v, 6) for v in case.label_spacing],
                "image_orientation": list(case.image_orientation),
                "label_orientation": list(case.label_orientation),
                "shape_match": case.shape_match,
                "spacing_match": case.spacing_match,
                "affine_match": case.affine_match,
            }
            for case in cases
        ],
    }


def percentile_zscore(volume: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(volume, [low, high])
    clipped = np.clip(volume, lo, hi)
    mean = clipped.mean()
    std = clipped.std()
    if std < 1.0e-6:
        return np.zeros_like(clipped, dtype=np.float32)
    return ((clipped - mean) / std).astype(np.float32)


def positive_voxels(mask: np.ndarray) -> int:
    return int(mask.sum())


def label_histogram(raw_codes: np.ndarray) -> dict[str, int]:
    uniques, counts = np.unique(raw_codes.astype(np.int16), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(uniques, counts, strict=True)}


def write_metadata_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    import csv

    rows = list(rows)
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_metadata_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
