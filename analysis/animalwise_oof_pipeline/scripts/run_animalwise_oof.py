from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from math import log
from pathlib import Path
from typing import Any, Optional, Union

import nibabel as nib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[3]


TIME_ORDER = {
    "D02": 2,
    "D09": 9,
    "D9": 9,
    "D28": 28,
    "W04": 28,
    "M01": 30,
    "W20": 140,
    "M05": 150,
    "M5": 150,
    "S1": 1001,
    "S2": 1002,
    "S3": 1003,
    "S4": 1004,
}


@dataclass
class CaseRecord:
    case_id: str
    cohort: str
    animal_id_strict: str
    animal_id_raw: str
    animal_family: str
    time_raw: str
    image_path: Path
    round0_label_path: Optional[Path]
    round1_label_path: Optional[Path]
    eval_label_path: Optional[Path]
    revised_status: str = "none"
    timepoint_pattern: str = ""
    fold: Optional[int] = None
    positive_voxels_round0: int = 0


@dataclass
class PreparedCase:
    record: CaseRecord
    image: np.ndarray
    target: np.ndarray
    eval_label: np.ndarray
    voxel_weight: np.ndarray
    case_weight: float
    positive_mask: np.ndarray


@dataclass
class NiftiVolume:
    data: np.ndarray
    affine: np.ndarray
    header: nib.Nifti1Header


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


def percentile_zscore(volume: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(volume, [low, high])
    clipped = np.clip(volume, lo, hi)
    mean = clipped.mean()
    std = clipped.std()
    if std < 1.0e-6:
        return np.zeros_like(clipped, dtype=np.float32)
    return ((clipped - mean) / std).astype(np.float32)


def build_soft_target(binary_mask: np.ndarray, oof_probability: np.ndarray, alpha: float) -> np.ndarray:
    target = (1.0 - alpha) * binary_mask.astype(np.float32) + alpha * oof_probability.astype(np.float32)
    return np.clip(target, 1.0e-4, 1.0 - 1.0e-4).astype(np.float32)


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-7, 1.0 - 1.0e-7)
    return (-clipped * np.log(clipped) - (1.0 - clipped) * np.log(1.0 - clipped)) / log(2)


def compute_uncertainty_from_target(soft_target: np.ndarray, alpha: float) -> np.ndarray:
    normalizer = float(binary_entropy(np.asarray([alpha], dtype=np.float32))[0])
    if normalizer <= 0.0:
        return np.zeros_like(soft_target, dtype=np.float32)
    return (binary_entropy(soft_target) / normalizer).astype(np.float32)


def build_voxel_weights(train_uncertainty: np.ndarray, reviewed: bool, floor_weight: float = 0.5) -> np.ndarray:
    if reviewed:
        return np.ones_like(train_uncertainty, dtype=np.float32)
    return np.clip(1.0 - 0.5 * train_uncertainty, floor_weight, 1.0).astype(np.float32)


def apply_tta(volume: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return volume
    if mode == "flip_x":
        return np.flip(volume, axis=0).copy()
    if mode == "flip_y":
        return np.flip(volume, axis=1).copy()
    if mode == "flip_xy":
        return np.flip(np.flip(volume, axis=0), axis=1).copy()
    raise ValueError(f"Unsupported TTA mode: {mode}")


def invert_tta(volume: np.ndarray, mode: str) -> np.ndarray:
    return apply_tta(volume, mode)


def iter_sliding_windows(shape, patch_size, overlap):
    strides = [max(1, int(size * (1.0 - overlap))) for size in patch_size]
    axes = []
    for dim, patch, stride in zip(shape, patch_size, strides):
        if dim <= patch:
            axes.append([0])
            continue
        starts = list(range(0, dim - patch + 1, stride))
        if starts[-1] != dim - patch:
            starts.append(dim - patch)
        axes.append(starts)
    for start_x in axes[0]:
        for start_y in axes[1]:
            for start_z in axes[2]:
                yield (
                    slice(start_x, start_x + patch_size[0]),
                    slice(start_y, start_y + patch_size[1]),
                    slice(start_z, start_z + patch_size[2]),
                )


def sliding_window_predict(model: torch.nn.Module, volume: np.ndarray, device: torch.device, patch_size, overlap: float) -> np.ndarray:
    model.eval()
    accum = np.zeros(volume.shape, dtype=np.float32)
    counts = np.zeros(volume.shape, dtype=np.float32)
    with torch.no_grad():
        for slices in iter_sliding_windows(volume.shape, patch_size, overlap):
            patch = volume[slices][None, None]
            patch_tensor = torch.from_numpy(patch).to(device=device, dtype=torch.float32)
            logits = model(patch_tensor)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            accum[slices] += probs
            counts[slices] += 1.0
    counts[counts == 0] = 1.0
    return accum / counts


def threshold_probability(probability: np.ndarray, threshold: float) -> np.ndarray:
    return (probability >= float(threshold)).astype(np.uint8)


def remove_small_components(binary_mask: np.ndarray, min_component_voxels: int) -> np.ndarray:
    if min_component_voxels <= 1:
        return binary_mask.astype(np.uint8, copy=False)
    labeled, num = ndimage.label(binary_mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
    if num == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    component_sizes = ndimage.sum(binary_mask.astype(np.uint8), labeled, index=np.arange(1, num + 1))
    keep = np.zeros(num + 1, dtype=bool)
    for idx, size in enumerate(component_sizes, start=1):
        if int(size) >= min_component_voxels:
            keep[idx] = True
    return keep[labeled].astype(np.uint8)


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(binary_mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
    if num == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    component_sizes = ndimage.sum(binary_mask.astype(np.uint8), labeled, index=np.arange(1, num + 1))
    largest_index = int(np.argmax(component_sizes)) + 1
    return (labeled == largest_index).astype(np.uint8)


def postprocess_probability_map(
    probability: np.ndarray,
    threshold: float,
    min_component_voxels: int = 0,
    largest_only: bool = False,
) -> np.ndarray:
    binary_mask = threshold_probability(probability, threshold)
    binary_mask = remove_small_components(binary_mask, min_component_voxels)
    if largest_only:
        binary_mask = keep_largest_component(binary_mask)
    return binary_mask.astype(np.uint8, copy=False)


def _norm(num_channels: int) -> nn.Module:
    return nn.InstanceNorm3d(num_channels, affine=True)


def _act() -> nn.Module:
    return nn.LeakyReLU(0.01, inplace=True)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _norm(out_channels)
        self.act1 = _act()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _norm(out_channels)
        self.act2 = _act()
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                _norm(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act2(out + identity)
        return out


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        self.downsample = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
                _norm(out_channels),
                _act(),
            )
            if downsample
            else None
        )
        block_in = out_channels if downsample else in_channels
        self.block1 = ResidualBlock3D(block_in, out_channels)
        self.block2 = ResidualBlock3D(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block1 = ResidualBlock3D(out_channels + skip_channels, out_channels)
        self.block2 = ResidualBlock3D(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff = [skip.shape[idx] - x.shape[idx] for idx in range(2, 5)]
        if any(diff):
            x = F.pad(
                x,
                [
                    max(diff[2] // 2, 0),
                    max(diff[2] - diff[2] // 2, 0),
                    max(diff[1] // 2, 0),
                    max(diff[1] - diff[1] // 2, 0),
                    max(diff[0] // 2, 0),
                    max(diff[0] - diff[0] // 2, 0),
                ],
            )
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return x


class ResidualUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        stage_channels=(32, 64, 96, 160, 256, 320),
        dropout_bottleneck: float = 0.1,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 6:
            raise ValueError("Expected 6 stage channels for the 6-stage encoder")
        self.stem = EncoderStage(in_channels, stage_channels[0], downsample=False)
        self.encoders = nn.ModuleList(
            [
                EncoderStage(stage_channels[0], stage_channels[1], downsample=True),
                EncoderStage(stage_channels[1], stage_channels[2], downsample=True),
                EncoderStage(stage_channels[2], stage_channels[3], downsample=True),
                EncoderStage(stage_channels[3], stage_channels[4], downsample=True),
                EncoderStage(stage_channels[4], stage_channels[5], downsample=True),
            ]
        )
        self.dropout = nn.Dropout3d(dropout_bottleneck) if dropout_bottleneck > 0 else nn.Identity()
        self.decoders = nn.ModuleList(
            [
                DecoderStage(stage_channels[5], stage_channels[4], stage_channels[4]),
                DecoderStage(stage_channels[4], stage_channels[3], stage_channels[3]),
                DecoderStage(stage_channels[3], stage_channels[2], stage_channels[2]),
                DecoderStage(stage_channels[2], stage_channels[1], stage_channels[1]),
                DecoderStage(stage_channels[1], stage_channels[0], stage_channels[0]),
            ]
        )
        self.head = nn.Conv3d(stage_channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for encoder in self.encoders:
            skips.append(encoder(skips[-1]))
        x = self.dropout(skips[-1])
        x = self.decoders[0](x, skips[-2])
        x = self.decoders[1](x, skips[-3])
        x = self.decoders[2](x, skips[-4])
        x = self.decoders[3](x, skips[-5])
        x = self.decoders[4](x, skips[-6])
        return self.head(x)


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


class PatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cases: list[PreparedCase],
        patch_size: tuple[int, int, int],
        patches_per_case: int,
        positive_patch_probability: float,
        seed: int,
        augment: bool,
    ) -> None:
        self.cases = cases
        self.patch_size = patch_size
        self.patches_per_case = int(patches_per_case)
        self.positive_patch_probability = float(positive_patch_probability)
        self.rng = np.random.default_rng(seed)
        self.augment = augment

    def __len__(self) -> int:
        return max(1, len(self.cases) * self.patches_per_case)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case = self.cases[index % len(self.cases)]
        slices = self._sample_slices(case)
        image = case.image[slices].astype(np.float32, copy=True)
        target = case.target[slices].astype(np.float32, copy=True)
        voxel_weight = case.voxel_weight[slices].astype(np.float32, copy=True)
        if self.augment:
            image, target, voxel_weight = self._augment(image, target, voxel_weight)
        return {
            "image": torch.from_numpy(image[None]),
            "target": torch.from_numpy(target[None]),
            "voxel_weight": torch.from_numpy(voxel_weight[None]),
            "case_weight": torch.tensor(case.case_weight, dtype=torch.float32),
        }

    def _sample_slices(self, case: PreparedCase) -> tuple[slice, slice, slice]:
        shape = case.image.shape
        use_positive = self.rng.random() < self.positive_patch_probability and bool(case.positive_mask.any())
        if use_positive:
            indices = np.argwhere(case.positive_mask > 0)
            center = indices[int(self.rng.integers(0, len(indices)))]
        else:
            center = np.asarray([self.rng.integers(0, dim) for dim in shape], dtype=np.int64)
        starts = []
        for axis, (dim, patch) in enumerate(zip(shape, self.patch_size)):
            if dim <= patch:
                starts.append(0)
                continue
            start = int(center[axis] - patch // 2)
            start = max(0, min(start, dim - patch))
            starts.append(start)
        return tuple(slice(starts[axis], starts[axis] + self.patch_size[axis]) for axis in range(3))

    def _augment(self, image: np.ndarray, target: np.ndarray, voxel_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for axis in (0, 1):
            if self.rng.random() < 0.5:
                image = np.flip(image, axis=axis).copy()
                target = np.flip(target, axis=axis).copy()
                voxel_weight = np.flip(voxel_weight, axis=axis).copy()
        scale = float(self.rng.uniform(0.9, 1.1))
        shift = float(self.rng.uniform(-0.1, 0.1))
        noise = self.rng.normal(0.0, 0.01, size=image.shape).astype(np.float32)
        image = (image * scale + shift + noise).astype(np.float32)
        return image, target, voxel_weight


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def strip_nii_gz(path: Union[Path, str]) -> str:
    name = Path(path).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def resolve_path(value: Optional[Union[str, Path]], base: Path = PROJECT_ROOT) -> Optional[Path]:
    if value is None:
        return None
    text = os.path.expandvars(str(value))
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg.setdefault("paths", {})
    cfg.setdefault("split", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("inference", {})
    cfg.setdefault("stages", {})
    cfg.setdefault("limits", {})
    return cfg


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Optional[list[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def update_state(run_dir: Path, **updates: Any) -> None:
    path = run_dir / "run_state.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    else:
        state = {}
    state.update(updates)
    state["updated_at"] = timestamp()
    write_json(path, state)


def git_hash() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def command_string() -> str:
    return " ".join([sys.executable, *sys.argv])


def launch_background(args: argparse.Namespace, config_path: Path, cfg: dict[str, Any]) -> None:
    run_dir = make_run_dir(cfg, args.run_dir, args.resume)
    ensure_run_dirs(run_dir)
    resolved_config = run_dir / "config_resolved.yaml"
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
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=stdout,
            stderr=stderr,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    write_json(
        run_dir / "run_state.json",
        {
            "status": "running",
            "current_stage": "launcher",
            "started_at": timestamp(),
            "updated_at": timestamp(),
            "pid": process.pid,
            "command": " ".join(cmd),
            "config": str(config_path.resolve()),
            "git_hash": git_hash(),
            "completed_stages": [],
            "failed_stage": None,
        },
    )
    print(f"Started background animal-wise OOF run: {run_dir}")
    print(f"PID: {process.pid}")
    print(f"Logs: {stdout_path} | {stderr_path}")


def make_run_dir(cfg: dict[str, Any], explicit_run_dir: Optional[str], resume: Optional[str]) -> Path:
    if resume:
        return resolve_path(resume)  # type: ignore[return-value]
    if explicit_run_dir:
        return resolve_path(explicit_run_dir)  # type: ignore[return-value]
    run_root = resolve_path(cfg["paths"].get("run_root", "analysis/animalwise_oof_pipeline/runs"))
    name = str(cfg.get("experiment_name", "animalwise_oof"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(run_root / f"{stamp}_{name}")  # type: ignore[operator]


def ensure_run_dirs(run_dir: Path) -> None:
    for rel in ["metadata", "splits", "checkpoints", "oof", "predictions", "metrics", "logs"]:
        ensure_dir(run_dir / rel)


def parse_epi_animal(case_id: str) -> tuple[str, str, str]:
    mhr = re.search(r"MHR_\d+", case_id)
    if mhr:
        return mhr.group(0), mhr.group(0), "MHR"
    rat = re.search(r"Rat\d+", case_id)
    if rat:
        return rat.group(0), rat.group(0), "B4C_Rat"
    return case_id, case_id, "unknown"


def parse_epi_time(case_id: str) -> str:
    for pattern in [r"_(D02|D09|D28|M01|M05|W04|W20)_", r"_M_(D02|D09|D28|M01|M05|W04|W20)_"]:
        match = re.search(pattern, case_id)
        if match:
            return match.group(1).upper()
    match = re.search(r"_M_1_(\d)_", case_id)
    if match:
        return f"S{match.group(1)}"
    return "unknown"


def parse_aramra_time(case_id: str) -> str:
    if re.search(r"_(D9|9D|9d)_", case_id):
        return "D9"
    if re.search(r"_(M5|5M)_", case_id):
        return "M5"
    return "unknown"


def parse_aramra_animal(case_id: str) -> tuple[str, str]:
    match = re.search(r"ARAMRA002_([^_]+)_", case_id)
    raw = match.group(1) if match else case_id
    token = raw
    if token.startswith("R") and len(token) > 1 and token[1].isdigit():
        token = token[1:]
    if token.startswith("A") and len(token) > 1 and token[1:].isdigit():
        return raw, token
    digit_match = re.match(r"(\d+)", token)
    if digit_match:
        return raw, f"A{digit_match.group(1)}"
    return raw, token


def normalize_aramra_base(path: Path) -> str:
    stem = strip_nii_gz(path)
    for suffix in ["_meanMag_SEG", "_mag_SEG", "_SEG", "_meanMag", "_mag", "_0000"]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def aramra_case_id_from_standard_image(path: Path) -> str:
    stem = strip_nii_gz(path)
    return stem[:-5] if stem.endswith("_0000") else stem


def read_review_status(workspace: Path) -> dict[str, str]:
    path = workspace / "reports" / "round_1" / "review_stats.csv"
    status: dict[str, str] = {}
    if not path.exists():
        return status
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = row.get("case_id", "")
            role = row.get("role", "") or "none"
            if case_id:
                status[case_id] = role
    return status


def scan_epibios_cases(cfg: dict[str, Any], logger: TeeLogger) -> list[CaseRecord]:
    workspace = resolve_path(cfg["paths"]["source_workspace"])
    data_root = resolve_path(cfg["paths"]["epibios_data_root"])
    assert workspace is not None and data_root is not None
    r0_dir = workspace / "artifacts" / "labels" / "binary" / "round_0"
    r1_dir = workspace / "artifacts" / "labels" / "binary" / "round_1"
    image_dir = data_root / "imagesTr"
    if not r0_dir.exists() or not r1_dir.exists():
        raise FileNotFoundError(f"Missing round labels: {r0_dir} / {r1_dir}")
    review_status = read_review_status(workspace)
    records: list[CaseRecord] = []
    for r0_label in sorted(r0_dir.glob("*.nii.gz")):
        case_id = strip_nii_gz(r0_label)
        r1_label = r1_dir / f"{case_id}.nii.gz"
        image = image_dir / f"{case_id}_0000.nii.gz"
        if not r1_label.exists():
            raise FileNotFoundError(f"Missing round1 binary label for {case_id}: {r1_label}")
        if not image.exists():
            raise FileNotFoundError(f"Missing EpiBios image for {case_id}: {image}")
        animal_raw, animal_strict, family = parse_epi_animal(case_id)
        records.append(
            CaseRecord(
                case_id=case_id,
                cohort="EpiBios",
                animal_id_strict=animal_strict,
                animal_id_raw=animal_raw,
                animal_family=family,
                time_raw=parse_epi_time(case_id),
                image_path=image.resolve(),
                round0_label_path=r0_label.resolve(),
                round1_label_path=r1_label.resolve(),
                eval_label_path=None,
                revised_status=review_status.get(case_id, "none"),
            )
        )
    apply_timepoint_patterns(records)
    limit = cfg.get("limits", {}).get("epibios_animals")
    if limit:
        keep_animals = sorted({record.animal_id_strict for record in records})[: int(limit)]
        records = [record for record in records if record.animal_id_strict in keep_animals]
        logger.log(f"limited EpiBios animals to {len(keep_animals)} for smoke/test")
    logger.log(f"loaded EpiBios cases={len(records)} animals={len({r.animal_id_strict for r in records})}")
    return records


def scan_aramra_cases(cfg: dict[str, Any], logger: TeeLogger) -> list[CaseRecord]:
    aramra_root = resolve_path(cfg["paths"].get("aramra_root"))
    if aramra_root is None or not aramra_root.exists():
        logger.log(f"ARAMRA root missing; skipping ARAMRA scan: {aramra_root}")
        return []
    images_dir = resolve_path(cfg["paths"].get("aramra_images_dir")) or (aramra_root / "imagesTs")
    labels_dir = resolve_path(cfg["paths"].get("aramra_labels_dir")) or (aramra_root / "labelsTs")
    if images_dir.exists() and labels_dir.exists():
        records: list[CaseRecord] = []
        missing_labels: list[str] = []
        image_paths = sorted(path for path in images_dir.glob("*.nii.gz") if not path.name.startswith("._"))
        for image_path in image_paths:
            case_id = aramra_case_id_from_standard_image(image_path)
            label_path = labels_dir / f"{case_id}.nii.gz"
            if not label_path.exists():
                missing_labels.append(case_id)
                continue
            animal_raw, animal_strict = parse_aramra_animal(case_id)
            records.append(
                CaseRecord(
                    case_id=case_id,
                    cohort="ARAMRA002",
                    animal_id_strict=animal_strict,
                    animal_id_raw=animal_raw,
                    animal_family="ARAMRA002",
                    time_raw=parse_aramra_time(case_id),
                    image_path=image_path.resolve(),
                    round0_label_path=None,
                    round1_label_path=None,
                    eval_label_path=label_path.resolve(),
                    revised_status="target_eval_only",
                )
            )
        label_case_ids = {
            strip_nii_gz(path)
            for path in labels_dir.glob("*.nii.gz")
            if not path.name.startswith("._")
        }
        image_case_ids = {aramra_case_id_from_standard_image(path) for path in image_paths}
        unmatched_labels = sorted(label_case_ids - image_case_ids)
        if missing_labels or unmatched_labels:
            logger.log(
                "standard ARAMRA layout unmatched files: "
                f"missing_labels_for_images={len(missing_labels)} "
                f"labels_without_images={len(unmatched_labels)}"
            )
        apply_timepoint_patterns(records)
        limit = cfg.get("limits", {}).get("aramra_cases")
        if limit:
            records = records[: int(limit)]
            apply_timepoint_patterns(records)
            logger.log(f"limited ARAMRA cases to {len(records)} for smoke/test")
        logger.log(
            f"loaded ARAMRA standardized cases={len(records)} "
            f"animals={len({r.animal_id_strict for r in records})} "
            f"from images={images_dir} labels={labels_dir}"
        )
        return records

    logger.log(f"standard ARAMRA imagesTs/labelsTs not found under {aramra_root}; falling back to recursive scan")
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    for path in sorted(aramra_root.rglob("*.nii.gz")):
        name = path.name
        if name.startswith("._"):
            continue
        if "ARAMRA002" not in name:
            continue
        base = normalize_aramra_base(path)
        stem = strip_nii_gz(path)
        if "SEG" in stem:
            labels.setdefault(base, path.resolve())
        elif "QSM" not in stem and ("meanMag" in stem or "_mag" in stem or stem.endswith("_0000")):
            images.setdefault(base, path.resolve())
    records: list[CaseRecord] = []
    for case_id in sorted(set(images) & set(labels)):
        animal_raw, animal_strict = parse_aramra_animal(case_id)
        records.append(
            CaseRecord(
                case_id=case_id,
                cohort="ARAMRA002",
                animal_id_strict=animal_strict,
                animal_id_raw=animal_raw,
                animal_family="ARAMRA002",
                time_raw=parse_aramra_time(case_id),
                image_path=images[case_id],
                round0_label_path=None,
                round1_label_path=None,
                eval_label_path=labels[case_id],
                revised_status="target_eval_only",
            )
        )
    apply_timepoint_patterns(records)
    limit = cfg.get("limits", {}).get("aramra_cases")
    if limit:
        records = records[: int(limit)]
        apply_timepoint_patterns(records)
        logger.log(f"limited ARAMRA cases to {len(records)} for smoke/test")
    logger.log(f"loaded ARAMRA labeled cases={len(records)} animals={len({r.animal_id_strict for r in records})}")
    return records


def apply_timepoint_patterns(records: list[CaseRecord]) -> None:
    by_animal: dict[str, list[CaseRecord]] = defaultdict(list)
    for record in records:
        by_animal[record.animal_id_strict].append(record)
    for animal_records in by_animal.values():
        times = sorted([record.time_raw for record in animal_records], key=lambda item: (TIME_ORDER.get(item, 9999), item))
        pattern = " / ".join(times)
        for record in animal_records:
            record.timepoint_pattern = pattern


def load_binary_label(path: Path) -> np.ndarray:
    data = load_nifti(path).data
    return (data > 0).astype(np.uint8)


def fill_positive_voxels(records: list[CaseRecord]) -> None:
    for record in records:
        if record.round0_label_path is None:
            continue
        record.positive_voxels_round0 = int(load_binary_label(record.round0_label_path).sum())


def make_animalwise_folds(records: list[CaseRecord], folds: int, seed: int) -> dict[str, int]:
    fill_positive_voxels(records)
    animals = sorted({record.animal_id_strict for record in records})
    family_keys = sorted({record.animal_family for record in records})
    time_keys = sorted({record.time_raw for record in records})
    review_keys = sorted({record.revised_status for record in records})
    animal_features: dict[str, dict[str, Any]] = {}
    for animal in animals:
        items = [record for record in records if record.animal_id_strict == animal]
        animal_features[animal] = {
            "cases": len(items),
            "pos": sum(record.positive_voxels_round0 for record in items),
            "family": Counter(record.animal_family for record in items),
            "time": Counter(record.time_raw for record in items),
            "review": Counter(record.revised_status for record in items),
        }
    totals = {
        "animals": len(animals),
        "cases": len(records),
        "pos": sum(features["pos"] for features in animal_features.values()),
        "family": Counter(),
        "time": Counter(),
        "review": Counter(),
    }
    for features in animal_features.values():
        totals["family"].update(features["family"])
        totals["time"].update(features["time"])
        totals["review"].update(features["review"])
    fold_stats = [
        {"animals": 0, "cases": 0, "pos": 0, "family": Counter(), "time": Counter(), "review": Counter()}
        for _ in range(folds)
    ]
    rng = np.random.default_rng(seed)
    ordered = sorted(
        animals,
        key=lambda animal: (
            -animal_features[animal]["pos"],
            -animal_features[animal]["cases"],
            rng.random(),
        ),
    )
    assignment: dict[str, int] = {}
    for animal in ordered:
        best_fold = min(range(folds), key=lambda fold_idx: fold_assignment_score(fold_stats, totals, animal_features[animal], fold_idx, folds, family_keys, time_keys, review_keys))
        assignment[animal] = best_fold + 1
        add_features(fold_stats[best_fold], animal_features[animal])
    return assignment


def add_features(stats: dict[str, Any], features: dict[str, Any]) -> None:
    stats["animals"] += 1
    stats["cases"] += int(features["cases"])
    stats["pos"] += int(features["pos"])
    stats["family"].update(features["family"])
    stats["time"].update(features["time"])
    stats["review"].update(features["review"])


def fold_assignment_score(
    fold_stats: list[dict[str, Any]],
    totals: dict[str, Any],
    features: dict[str, Any],
    fold_idx: int,
    folds: int,
    family_keys: list[str],
    time_keys: list[str],
    review_keys: list[str],
) -> float:
    trial = []
    for idx, stats in enumerate(fold_stats):
        copied = {
            "animals": stats["animals"],
            "cases": stats["cases"],
            "pos": stats["pos"],
            "family": Counter(stats["family"]),
            "time": Counter(stats["time"]),
            "review": Counter(stats["review"]),
        }
        if idx == fold_idx:
            add_features(copied, features)
        trial.append(copied)
    score = 0.0
    for stats in trial:
        score += normalized_abs(stats["animals"], totals["animals"] / folds, 1.0) * 2.0
        score += normalized_abs(stats["cases"], totals["cases"] / folds, 1.0) * 2.0
        score += normalized_abs(stats["pos"], totals["pos"] / folds, 1.0)
        for key in family_keys:
            score += normalized_abs(stats["family"][key], totals["family"][key] / folds, 1.0) * 0.5
        for key in time_keys:
            score += normalized_abs(stats["time"][key], totals["time"][key] / folds, 1.0) * 0.25
        for key in review_keys:
            score += normalized_abs(stats["review"][key], totals["review"][key] / folds, 1.0) * 0.5
    return score


def normalized_abs(value: float, target: float, floor: float) -> float:
    return abs(float(value) - float(target)) / max(abs(float(target)), floor)


def write_metadata_and_split(run_dir: Path, epi_cases: list[CaseRecord], ar_cases: list[CaseRecord], folds: int, seed: int) -> None:
    fold_map = make_animalwise_folds(epi_cases, folds, seed)
    for record in epi_cases:
        record.fold = fold_map[record.animal_id_strict]
    metadata_rows = [record_to_row(record) for record in [*epi_cases, *ar_cases]]
    write_csv(run_dir / "metadata" / "metadata_master.csv", metadata_rows)
    write_csv(run_dir / "metadata" / "epibios_cases.csv", [record_to_row(record) for record in epi_cases])
    write_csv(run_dir / "metadata" / "aramra_cases.csv", [record_to_row(record) for record in ar_cases])
    split_rows = []
    for animal, fold in sorted(fold_map.items(), key=lambda item: (item[1], item[0])):
        items = [record for record in epi_cases if record.animal_id_strict == animal]
        split_rows.append(
            {
                "animal_id_strict": animal,
                "animalwise_fold": fold,
                "num_cases": len(items),
                "animal_family": items[0].animal_family if items else "",
                "timepoint_pattern": items[0].timepoint_pattern if items else "",
                "total_round0_positive_voxels": sum(record.positive_voxels_round0 for record in items),
                "reviewed_cases": sum(record.revised_status != "none" for record in items),
            }
        )
    write_csv(run_dir / "splits" / "epibios_animalwise_folds.csv", split_rows)
    write_csv(run_dir / "splits" / "aramra_eval_manifest.csv", [record_to_row(record) for record in ar_cases])
    write_json(run_dir / "splits" / "split_integrity_report.json", split_integrity(epi_cases, ar_cases, folds))


def record_to_row(record: CaseRecord) -> dict[str, Any]:
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
        "positive_voxels_round0": record.positive_voxels_round0,
        "image_path": str(record.image_path),
        "round0_label_path": str(record.round0_label_path or ""),
        "round1_label_path": str(record.round1_label_path or ""),
        "eval_label_path": str(record.eval_label_path or ""),
    }


def split_integrity(epi_cases: list[CaseRecord], ar_cases: list[CaseRecord], folds: int) -> dict[str, Any]:
    fold_animals = {fold: {record.animal_id_strict for record in epi_cases if record.fold == fold} for fold in range(1, folds + 1)}
    overlaps = []
    for a, b in combinations(range(1, folds + 1), 2):
        overlap = sorted(fold_animals[a] & fold_animals[b])
        if overlap:
            overlaps.append({"fold_a": a, "fold_b": b, "overlap": overlap})
    fold_summary = []
    for fold in range(1, folds + 1):
        items = [record for record in epi_cases if record.fold == fold]
        fold_summary.append(
            {
                "fold": fold,
                "num_animals": len({record.animal_id_strict for record in items}),
                "num_cases": len(items),
                "positive_voxels": sum(record.positive_voxels_round0 for record in items),
                "families": dict(Counter(record.animal_family for record in items)),
                "timepoints": dict(Counter(record.time_raw for record in items)),
                "revised_status": dict(Counter(record.revised_status for record in items)),
            }
        )
    return {
        "status": "pass" if not overlaps else "fail",
        "num_epibios_cases": len(epi_cases),
        "num_epibios_animals": len({record.animal_id_strict for record in epi_cases}),
        "num_aramra_cases": len(ar_cases),
        "num_aramra_animals": len({record.animal_id_strict for record in ar_cases}),
        "animal_overlap_between_folds": overlaps,
        "aramra_used_for_training": False,
        "fold_summary": fold_summary,
    }


def make_model(cfg: dict[str, Any]) -> ResidualUNet3D:
    model_cfg = cfg["model"]
    return ResidualUNet3D(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        stage_channels=tuple(int(v) for v in model_cfg.get("stage_channels", [32, 64, 96, 160, 256, 320])),
        dropout_bottleneck=float(model_cfg.get("dropout_bottleneck", 0.1)),
    )


def select_device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg["training"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def prepare_case(record: CaseRecord, stage: str, cfg: dict[str, Any], run_dir: Path) -> PreparedCase:
    image = percentile_zscore(load_nifti(record.image_path).data)
    if stage == "r0":
        assert record.round0_label_path is not None
        binary = load_binary_label(record.round0_label_path)
        target = binary.astype(np.float32)
        voxel_weight = np.ones_like(target, dtype=np.float32)
        case_weight = 1.0
    elif stage in {"r1", "r1_hard"}:
        assert record.round1_label_path is not None
        binary = load_binary_label(record.round1_label_path)
        if stage == "r1_hard":
            target = binary.astype(np.float32)
            voxel_weight = np.ones_like(target, dtype=np.float32)
        else:
            r0_prob = load_probability(run_dir / "oof" / "r0" / "probabilities" / f"{record.case_id}.npz")
            target = build_soft_target(binary, r0_prob, float(cfg["training"].get("alpha", 0.15)))
            uncertainty = compute_uncertainty_from_target(target, float(cfg["training"].get("alpha", 0.15)))
            voxel_weight = build_voxel_weights(
                uncertainty,
                reviewed=record.revised_status != "none",
                floor_weight=float(cfg["training"].get("voxel_weight_floor", 0.5)),
            )
        case_weight = float(cfg["training"].get("case_weight_reviewed", 2.0) if record.revised_status != "none" else cfg["training"].get("case_weight_unreviewed", 1.0))
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    return PreparedCase(
        record=record,
        image=image,
        target=target.astype(np.float32),
        eval_label=binary.astype(np.uint8),
        voxel_weight=voxel_weight.astype(np.float32),
        case_weight=case_weight,
        positive_mask=binary.astype(np.uint8),
    )


def load_probability(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        if "probability" in payload:
            return payload["probability"].astype(np.float32)
        return payload["s"].astype(np.float32)


def protocol_loss(logits: torch.Tensor, target: torch.Tensor, voxel_weight: torch.Tensor, case_weight: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    wbce = (bce * voxel_weight).mean(dim=(1, 2, 3, 4))
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3, 4))
    denom = probs.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    sdice = 1.0 - (2.0 * inter + 1.0e-6) / (denom + 1.0e-6)
    return (case_weight * (0.5 * wbce + 0.5 * sdice)).mean()


def split_train_val(train_records: list[CaseRecord], cfg: dict[str, Any], outer_fold: int) -> tuple[list[CaseRecord], list[CaseRecord]]:
    val_cfg = cfg["training"].get("validation", {})
    if not bool(val_cfg.get("enabled", True)) or len(train_records) <= 1:
        return train_records, []
    animals = sorted({record.animal_id_strict for record in train_records})
    min_cases = int(val_cfg.get("min_cases", 1))
    target_cases = max(min_cases, int(round(len(train_records) * float(val_cfg.get("fraction", 0.125)))))
    selected_animals: list[str] = []
    selected_cases = 0
    ordered = sorted(animals, key=lambda animal: ((hash((animal, outer_fold)) % 1000000), animal))
    for animal in ordered:
        selected_animals.append(animal)
        selected_cases += sum(record.animal_id_strict == animal for record in train_records)
        if selected_cases >= target_cases and len(selected_animals) < len(animals):
            break
    val_animals = set(selected_animals)
    train = [record for record in train_records if record.animal_id_strict not in val_animals]
    val = [record for record in train_records if record.animal_id_strict in val_animals]
    if not train:
        return train_records, []
    return train, val


def train_stage(stage: str, epi_cases: list[CaseRecord], cfg: dict[str, Any], run_dir: Path, logger: TeeLogger) -> None:
    folds = int(cfg["split"].get("folds", 5))
    stage_name = stage
    if stage == "r1_hard":
        phase_cfg = cfg["training"]["round1"]
        init_stage = "r0"
    else:
        phase_cfg = cfg["training"]["round0" if stage == "r0" else "round1"]
        init_stage = None if stage == "r0" else "r0"
    device = select_device(cfg)
    stage_metrics: list[dict[str, Any]] = []
    for fold in range(1, folds + 1):
        update_state(run_dir, current_stage=f"{stage_name}_fold_{fold}_training")
        fold_log = TeeLogger(run_dir / "logs" / f"train_{stage_name}_fold_{fold}.log")
        try:
            train_records = [record for record in epi_cases if record.fold != fold]
            holdout_records = [record for record in epi_cases if record.fold == fold]
            train_records, val_records = split_train_val(train_records, cfg, fold)
            fold_log.log(f"start {stage_name} fold={fold} train_cases={len(train_records)} val_cases={len(val_records)} holdout_cases={len(holdout_records)}")
            train_prepared = [prepare_case(record, stage, cfg, run_dir) for record in train_records]
            val_prepared = [prepare_case(record, stage, cfg, run_dir) for record in val_records]
            model = make_model(cfg).to(device)
            if init_stage is not None:
                init_path = run_dir / "checkpoints" / init_stage / f"fold_{fold}_best.pt"
                load_checkpoint(model, init_path, device)
                fold_log.log(f"loaded init checkpoint {init_path}")
            best_path, last_path, train_rows = fit_model(model, train_prepared, val_prepared, cfg, phase_cfg, device, fold_log, stage_name, fold)
            ensure_dir(run_dir / "checkpoints" / stage_name)
            shutil.copy2(best_path, run_dir / "checkpoints" / stage_name / f"fold_{fold}_best.pt")
            shutil.copy2(last_path, run_dir / "checkpoints" / stage_name / f"fold_{fold}_last.pt")
            write_csv(run_dir / "metrics" / stage_name / f"fold_{fold}_train.csv", train_rows)
            load_checkpoint(model, run_dir / "checkpoints" / stage_name / f"fold_{fold}_best.pt", device)
            oof_rows = predict_records(model, holdout_records, stage, cfg, run_dir, device, output_root=run_dir / "oof" / stage_name, fold=fold, logger=fold_log)
            stage_metrics.extend(oof_rows)
            fold_log.log(f"complete {stage_name} fold={fold}")
        finally:
            fold_log.close()
    write_stage_metrics(run_dir, stage_name, stage_metrics)
    append_completed_stage(run_dir, stage_name)
    logger.log(f"completed stage {stage_name}")


def fit_model(
    model: torch.nn.Module,
    train_cases: list[PreparedCase],
    val_cases: list[PreparedCase],
    cfg: dict[str, Any],
    phase_cfg: dict[str, Any],
    device: torch.device,
    logger: TeeLogger,
    stage: str,
    fold: int,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    tmp_dir = ensure_dir(Path(cfg["_run_dir"]) / "checkpoints" / "_tmp" / stage)
    train_cfg = cfg["training"]
    dataset = PatchDataset(
        train_cases,
        patch_size=tuple(int(v) for v in train_cfg.get("patch_size", [128, 96, 64])),
        patches_per_case=int(train_cfg.get("patches_per_case", 2)),
        positive_patch_probability=float(train_cfg.get("positive_patch_probability", 0.5)),
        seed=int(cfg.get("seed", 20260524)) + fold,
        augment=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(phase_cfg["lr"]), weight_decay=float(phase_cfg.get("weight_decay", 1.0e-4)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(train_cfg.get("amp", True)))
    epochs = int(phase_cfg["epochs"])
    best_metric = float("-inf")
    best_epoch = 0
    rows: list[dict[str, Any]] = []
    best_path = tmp_dir / f"{stage}_fold_{fold}_best.pt"
    last_path = tmp_dir / f"{stage}_fold_{fold}_last.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            targets = batch["target"].to(device=device, dtype=torch.float32)
            weights = batch["voxel_weight"].to(device=device, dtype=torch.float32)
            case_weights = batch["case_weight"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda" and bool(train_cfg.get("amp", True))):
                logits = model(images)
                loss = protocol_loss(logits, targets, weights, case_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate_prepared_cases(model, val_cases, cfg, device) if val_cases else {}
        mean_loss = float(np.mean(losses)) if losses else 0.0
        metric = float(val_metrics.get("macro_dice_post_min2", -mean_loss))
        is_best = metric > best_metric
        if is_best:
            best_metric = metric
            best_epoch = epoch
            save_checkpoint(best_path, model, epoch, best_metric, cfg)
        save_checkpoint(last_path, model, epoch, metric, cfg)
        row = {
            "epoch": epoch,
            "mean_loss": mean_loss,
            "val_macro_dice_raw": val_metrics.get("macro_dice_raw", ""),
            "val_macro_dice_post_min2": val_metrics.get("macro_dice_post_min2", ""),
            "val_micro_dice_raw": val_metrics.get("micro_dice_raw", ""),
            "is_new_best": int(is_best),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        }
        rows.append(row)
        logger.log(f"epoch={epoch}/{epochs} loss={mean_loss:.6f} metric={metric:.6f} best={best_metric:.6f}")
    if not best_path.exists():
        save_checkpoint(best_path, model, epochs, best_metric, cfg)
    return best_path, last_path, rows


def save_checkpoint(path: Path, model: torch.nn.Module, epoch: int, metric: float, cfg: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save({"model_state": model.state_dict(), "epoch": epoch, "metric": metric, "model_config": cfg["model"]}, path)


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    payload = torch.load(path, map_location=device)
    state = payload.get("model_state", payload)
    model.load_state_dict(state)


def evaluate_prepared_cases(model: torch.nn.Module, cases: list[PreparedCase], cfg: dict[str, Any], device: torch.device) -> dict[str, float]:
    rows_raw = []
    rows_post = []
    for case in cases:
        prob = predict_volume(model, case.image, cfg, device)
        raw = (prob >= float(cfg["inference"].get("threshold", 0.5))).astype(np.uint8)
        post = postprocess_probability_map(prob, threshold=float(cfg["inference"].get("threshold", 0.5)), min_component_voxels=2, largest_only=False)
        rows_raw.append(metric_row(case.record, raw, case.eval_label, "raw", "validation"))
        rows_post.append(metric_row(case.record, post, case.eval_label, "post_min2", "validation"))
    raw = aggregate_rows(rows_raw)
    post = aggregate_rows(rows_post)
    return {
        "macro_dice_raw": raw["macro_dice"],
        "micro_dice_raw": raw["micro_dice"],
        "macro_dice_post_min2": post["macro_dice"],
    }


def predict_volume(model: torch.nn.Module, image: np.ndarray, cfg: dict[str, Any], device: torch.device) -> np.ndarray:
    modes = list(cfg["inference"].get("tta_modes", ["identity"]))
    patch_size = tuple(int(v) for v in cfg["inference"].get("patch_size", cfg["training"].get("patch_size", [128, 96, 64])))
    overlap = float(cfg["inference"].get("overlap", 0.5))
    probs = []
    for mode in modes:
        aug = apply_tta(image, mode)
        pred = sliding_window_predict(model, aug, device, patch_size, overlap)
        probs.append(invert_tta(pred, mode))
    return np.mean(np.stack(probs, axis=0), axis=0).astype(np.float32)


def predict_records(
    model: torch.nn.Module,
    records: list[CaseRecord],
    stage: str,
    cfg: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    output_root: Path,
    fold: Union[int, str],
    logger: TeeLogger,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    threshold = float(cfg["inference"].get("threshold", 0.5))
    post_values = [int(v) for v in cfg["inference"].get("postprocess_min_component_voxels", [2])]
    for idx, record in enumerate(records, start=1):
        image_vol = load_nifti(record.image_path)
        image = percentile_zscore(image_vol.data)
        label_path = record.round0_label_path if stage == "r0" else record.round1_label_path
        if record.cohort == "ARAMRA002":
            label_path = record.eval_label_path
        assert label_path is not None
        target = load_binary_label(label_path)
        prob = predict_volume(model, image, cfg, device)
        if bool(cfg["inference"].get("save_probabilities", True)):
            prob_path = output_root / "probabilities" / f"{record.case_id}.npz"
            ensure_dir(prob_path.parent)
            np.savez_compressed(prob_path, probability=prob, s=prob)
        raw = (prob >= threshold).astype(np.uint8)
        variants = {"raw": raw}
        for min_voxels in post_values:
            variants[f"post_min{min_voxels}"] = postprocess_probability_map(prob, threshold=threshold, min_component_voxels=min_voxels, largest_only=False)
        for variant, mask in variants.items():
            if bool(cfg["inference"].get("save_masks", True)):
                mask_path = output_root / f"masks_{variant}" / f"{record.case_id}.nii.gz"
                save_nifti(mask_path, mask, image_vol.affine, image_vol.header, np.uint8)
            rows.append(metric_row(record, mask, target, variant, stage, fold=fold))
        logger.log(f"predicted {stage} fold={fold} {idx}/{len(records)} case={record.case_id}")
    return rows


def metric_row(record: CaseRecord, pred: np.ndarray, target: np.ndarray, variant: str, stage: str, fold: Union[int, str] = "") -> dict[str, Any]:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    inter = int(np.logical_and(pred_bool, target_bool).sum())
    pred_sum = int(pred_bool.sum())
    target_sum = int(target_bool.sum())
    fp = int(np.logical_and(pred_bool, ~target_bool).sum())
    fn = int(np.logical_and(~pred_bool, target_bool).sum())
    return {
        "stage": stage,
        "fold": fold,
        "case_id": record.case_id,
        "cohort": record.cohort,
        "animal_id_strict": record.animal_id_strict,
        "animal_family": record.animal_family,
        "time_raw": record.time_raw,
        "timepoint_pattern": record.timepoint_pattern,
        "revised_status": record.revised_status,
        "postprocess_variant": variant,
        "dice": dice_from_counts(inter, pred_sum, target_sum),
        "intersection": inter,
        "pred_positive_voxels": pred_sum,
        "gt_positive_voxels": target_sum,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "absolute_volume_error_voxels": abs(pred_sum - target_sum),
        "signed_volume_error_voxels": pred_sum - target_sum,
        "hd95": hd95(pred_bool, target_bool),
        **component_metrics(pred_bool, target_bool),
    }


def dice_from_counts(intersection: int, pred_sum: int, target_sum: int) -> float:
    denom = pred_sum + target_sum
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


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


def surface(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask, ~eroded)


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


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    animal_values = []
    by_animal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_animal[str(row["animal_id_strict"])].append(row)
    for items in by_animal.values():
        animal_values.append(float(np.mean([float(item["dice"]) for item in items])))
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


def write_stage_metrics(run_dir: Path, stage: str, rows: list[dict[str, Any]]) -> None:
    write_csv(run_dir / "metrics" / f"{stage}_oof_case_metrics.csv", rows)
    fold_rows = []
    for (fold, variant), items in sorted(group_by(rows, ["fold", "postprocess_variant"]).items()):
        fold_rows.append({"fold": fold, "postprocess_variant": variant, **aggregate_rows(items)})
    write_csv(run_dir / "metrics" / f"{stage}_oof_fold_metrics.csv", fold_rows)
    summary = {"overall": {}}
    for variant, items in sorted(group_by(rows, ["postprocess_variant"]).items()):
        summary["overall"][variant[0] if isinstance(variant, tuple) else variant] = aggregate_rows(items)
    write_json(run_dir / "metrics" / f"{stage}_oof_summary.json", summary)


def group_by(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def append_completed_stage(run_dir: Path, stage: str) -> None:
    path = run_dir / "run_state.json"
    state = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    completed = list(state.get("completed_stages", []))
    if stage not in completed:
        completed.append(stage)
    update_state(run_dir, completed_stages=completed)


def evaluate_aramra(ar_cases: list[CaseRecord], cfg: dict[str, Any], run_dir: Path, logger: TeeLogger) -> None:
    if not ar_cases:
        logger.log("no ARAMRA cases found; skipping OOD evaluation")
        return
    device = select_device(cfg)
    all_rows: list[dict[str, Any]] = []
    for stage in ["r0", "r1"]:
        if not (run_dir / "checkpoints" / stage).exists():
            logger.log(f"missing checkpoints for {stage}; skipping ARAMRA eval")
            continue
        update_state(run_dir, current_stage=f"aramra_eval_{stage}")
        stage_rows: list[dict[str, Any]] = []
        models = []
        for fold in range(1, int(cfg["split"].get("folds", 5)) + 1):
            ckpt = run_dir / "checkpoints" / stage / f"fold_{fold}_best.pt"
            model = make_model(cfg).to(device)
            load_checkpoint(model, ckpt, device)
            model.eval()
            models.append(model)
        output_root = run_dir / "predictions" / "aramra" / stage
        eval_log = TeeLogger(run_dir / "logs" / f"eval_aramra_{stage}.log")
        try:
            for idx, record in enumerate(ar_cases, start=1):
                image_vol = load_nifti(record.image_path)
                image = percentile_zscore(image_vol.data)
                target = load_binary_label(record.eval_label_path)  # type: ignore[arg-type]
                fold_probs = [predict_volume(model, image, cfg, device) for model in models]
                prob = np.mean(np.stack(fold_probs, axis=0), axis=0).astype(np.float32)
                if bool(cfg["inference"].get("save_probabilities", True)):
                    prob_path = output_root / "probabilities" / f"{record.case_id}.npz"
                    ensure_dir(prob_path.parent)
                    np.savez_compressed(prob_path, probability=prob, s=prob)
                threshold = float(cfg["inference"].get("threshold", 0.5))
                variants = {"raw": (prob >= threshold).astype(np.uint8)}
                for min_voxels in [int(v) for v in cfg["inference"].get("postprocess_min_component_voxels", [2])]:
                    variants[f"post_min{min_voxels}"] = postprocess_probability_map(prob, threshold=threshold, min_component_voxels=min_voxels, largest_only=False)
                for variant, mask in variants.items():
                    if bool(cfg["inference"].get("save_masks", True)):
                        mask_path = output_root / f"masks_{variant}" / f"{record.case_id}.nii.gz"
                        save_nifti(mask_path, mask, image_vol.affine, image_vol.header, np.uint8)
                    row = metric_row(record, mask, target, variant, f"aramra_{stage}", fold="ensemble")
                    row["prediction_stage"] = stage
                    stage_rows.append(row)
                eval_log.log(f"predicted ARAMRA {stage} {idx}/{len(ar_cases)} case={record.case_id}")
        finally:
            eval_log.close()
        write_csv(run_dir / "metrics" / f"aramra_{stage}_case_metrics.csv", stage_rows)
        all_rows.extend(stage_rows)
    write_aramra_summaries(run_dir, all_rows)
    append_completed_stage(run_dir, "aramra_eval")
    logger.log("completed ARAMRA OOD evaluation")


def write_aramra_summaries(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    group_rows = []
    for (stage, variant), items in sorted(group_by(rows, ["prediction_stage", "postprocess_variant"]).items()):
        group_rows.append({"prediction_stage": stage, "postprocess_variant": variant, "group": "overall", **aggregate_rows(items)})
    for (stage, variant, time_raw), items in sorted(group_by(rows, ["prediction_stage", "postprocess_variant", "time_raw"]).items()):
        group_rows.append({"prediction_stage": stage, "postprocess_variant": variant, "group": f"time={time_raw}", **aggregate_rows(items)})
    for (stage, variant, pattern), items in sorted(group_by(rows, ["prediction_stage", "postprocess_variant", "timepoint_pattern"]).items()):
        group_rows.append({"prediction_stage": stage, "postprocess_variant": variant, "group": f"pattern={pattern}", **aggregate_rows(items)})
    write_csv(run_dir / "metrics" / "aramra_group_metrics.csv", group_rows)
    pair_rows = pair_metrics(rows)
    write_csv(run_dir / "metrics" / "aramra_pair_metrics.csv", pair_rows)
    summary = {
        "overall": {
            f"{stage}_{variant}": aggregate_rows(items)
            for (stage, variant), items in sorted(group_by(rows, ["prediction_stage", "postprocess_variant"]).items())
        },
        "pair_metrics": summarize_pair_rows(pair_rows),
    }
    write_json(run_dir / "metrics" / "aramra_summary.json", summary)


def pair_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_rows = [row for row in rows if row["postprocess_variant"] == "raw"]
    grouped = group_by(raw_rows, ["prediction_stage", "animal_id_strict"])
    out: list[dict[str, Any]] = []
    for (stage, animal), items in sorted(grouped.items()):
        d9 = [row for row in items if row["time_raw"] == "D9"]
        m5 = [row for row in items if row["time_raw"] == "M5"]
        for a, b in combinations(d9, 2):
            out.append(
                {
                    "analysis_type": "repeat_d9",
                    "prediction_stage": stage,
                    "animal_id_strict": animal,
                    "case_id_a": a["case_id"],
                    "case_id_b": b["case_id"],
                    "dice_a": a["dice"],
                    "dice_b": b["dice"],
                    "dice_delta": float(b["dice"]) - float(a["dice"]),
                }
            )
        for d9_row in d9:
            for m5_row in m5:
                out.append(
                    {
                        "analysis_type": "d9_m5_pair",
                        "prediction_stage": stage,
                        "animal_id_strict": animal,
                        "d9_case_id": d9_row["case_id"],
                        "m5_case_id": m5_row["case_id"],
                        "d9_dice": d9_row["dice"],
                        "m5_dice": m5_row["dice"],
                        "dice_delta_m5_minus_d9": float(m5_row["dice"]) - float(d9_row["dice"]),
                        "gt_volume_delta_m5_minus_d9": int(m5_row["gt_positive_voxels"]) - int(d9_row["gt_positive_voxels"]),
                        "pred_volume_delta_m5_minus_d9": int(m5_row["pred_positive_voxels"]) - int(d9_row["pred_positive_voxels"]),
                    }
                )
    return out


def summarize_pair_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for (stage, analysis_type), items in sorted(group_by(rows, ["prediction_stage", "analysis_type"]).items()):
        if analysis_type == "d9_m5_pair":
            deltas = [float(row["dice_delta_m5_minus_d9"]) for row in items]
        else:
            deltas = [float(row["dice_delta"]) for row in items]
        summary[f"{stage}_{analysis_type}"] = {
            "count": len(items),
            "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
            "median_delta": float(np.median(deltas)) if deltas else float("nan"),
        }
    return summary


def write_report(run_dir: Path, cfg: dict[str, Any]) -> None:
    lines = ["# Animal-wise OOF + ARAMRA OOD Report", ""]
    integrity_path = run_dir / "splits" / "split_integrity_report.json"
    if integrity_path.exists():
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        lines += [
            "## Split Integrity",
            "",
            f"- Status: `{integrity.get('status')}`",
            f"- EpiBios cases/animals: {integrity.get('num_epibios_cases')} / {integrity.get('num_epibios_animals')}",
            f"- ARAMRA cases/animals: {integrity.get('num_aramra_cases')} / {integrity.get('num_aramra_animals')}",
            f"- ARAMRA used for training: {integrity.get('aramra_used_for_training')}",
            "",
        ]
    lines += ["## EpiBios Animal-wise OOF", ""]
    for stage in ["r0", "r1", "r1_hard"]:
        path = run_dir / "metrics" / f"{stage}_oof_summary.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw = payload.get("overall", {}).get("raw", {})
            post = payload.get("overall", {}).get("post_min2", {})
            lines.append(f"- {stage}: raw macro Dice={fmt(raw.get('macro_dice'))}, animal macro Dice={fmt(raw.get('animal_macro_dice'))}; post_min2 macro Dice={fmt(post.get('macro_dice'))}")
    old_oof = old_workspace_oof_reference(cfg)
    if old_oof:
        lines += ["", "Old workspace scan-level OOF reference:"]
        for key, metrics in old_oof.items():
            lines.append(f"- {key}: raw macro Dice={fmt(metrics.get('macro_dice_raw'))}, post macro Dice={fmt(metrics.get('macro_dice_postprocessed'))}, raw micro Dice={fmt(metrics.get('micro_dice_raw'))}")
    lines += ["", "## ARAMRA OOD", ""]
    path = run_dir / "metrics" / "aramra_summary.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, metrics in payload.get("overall", {}).items():
            lines.append(f"- {key}: macro Dice={fmt(metrics.get('macro_dice'))}, animal macro Dice={fmt(metrics.get('animal_macro_dice'))}, micro Dice={fmt(metrics.get('micro_dice'))}, lesion-F1={fmt(metrics.get('mean_lesion_f1'))}, median HD95={fmt(metrics.get('median_hd95'))}")
        lines.append("")
        lines.append("Pair metrics:")
        for key, metrics in payload.get("pair_metrics", {}).items():
            lines.append(f"- {key}: n={metrics.get('count')}, mean delta={fmt(metrics.get('mean_delta'))}, median delta={fmt(metrics.get('median_delta'))}")
    old_aramra = old_aramra_reference()
    if old_aramra:
        lines += ["", "Old workspace ARAMRA reference:"]
        for key, metrics in old_aramra.items():
            lines.append(f"- {key}: macro Dice={fmt(metrics.get('macro_dice'))}, animal macro Dice={fmt(metrics.get('animal_macro_dice'))}, micro Dice={fmt(metrics.get('micro_dice'))}, lesion-F1={fmt(metrics.get('mean_lesion_f1'))}, median HD95={fmt(metrics.get('median_hd95'))}")
    lines += ["", "## Caveats", ""]
    lines += [
        "- This pipeline does not implement ARAMRA target adaptation.",
        "- R1 soft targets use the new animal-wise R0 OOF probabilities, not the old scan-level OOF.",
        "- Raw labels are not used directly as training targets; binary labels are used, with soft targets only for R1.",
        "- ARAMRA is read from a standardized imagesTs/labelsTs layout when available; recursive legacy scanning is only a fallback.",
        "- If server paths differ, update only the config paths before launching.",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def old_workspace_oof_reference(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    workspace = resolve_path(cfg["paths"].get("source_workspace"))
    if workspace is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for round_index, key in [(0, "old_scan_r0"), (1, "old_scan_r1")]:
        path = workspace / "reports" / f"round_{round_index}" / "oof_case_metrics.csv"
        if not path.exists():
            continue
        rows = read_csv_rows(path)
        if not rows:
            continue
        inter_raw = sum(int(float(row["intersection_raw"])) for row in rows)
        pred_raw = sum(int(float(row["pred_positive_voxels_raw"])) for row in rows)
        gt = sum(int(float(row["gt_positive_voxels"])) for row in rows)
        inter_post = sum(int(float(row["intersection_postprocessed"])) for row in rows)
        pred_post = sum(int(float(row["pred_positive_voxels_postprocessed"])) for row in rows)
        out[key] = {
            "macro_dice_raw": float(np.mean([float(row["dice_raw"]) for row in rows])),
            "macro_dice_postprocessed": float(np.mean([float(row["dice_postprocessed"]) for row in rows])),
            "micro_dice_raw": dice_from_counts(inter_raw, pred_raw, gt),
            "micro_dice_postprocessed": dice_from_counts(inter_post, pred_post, gt),
            "num_cases": len(rows),
        }
    return out


def old_aramra_reference() -> dict[str, dict[str, Any]]:
    path = PROJECT_ROOT / "analysis" / "workspace_v0_full_external_analysis" / "results" / "aramra_external_group_metrics.csv"
    if not path.exists():
        return {}
    rows = read_csv_rows(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("time_raw") or row.get("animal_family") or row.get("animal_timepoint_pattern"):
            continue
        if row.get("postprocess_variant") != "raw":
            continue
        round_index = row.get("prediction_round")
        key = f"old_aramra_r{round_index}_raw"
        out[key] = row
    return out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    try:
        if np.isnan(float(value)):
            return "NA"
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def run_pipeline(cfg: dict[str, Any], run_dir: Path) -> None:
    ensure_run_dirs(run_dir)
    cfg["_run_dir"] = str(run_dir)
    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
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
            failed_stage=None,
        )
        seed = int(cfg.get("seed", 20260524))
        torch.manual_seed(seed)
        np.random.seed(seed)
        logger.log(f"run_dir={run_dir}")
        logger.log(f"device={select_device(cfg)} torch_cuda={torch.cuda.is_available()}")
        epi_cases = scan_epibios_cases(cfg, logger)
        ar_cases = scan_aramra_cases(cfg, logger)
        update_state(run_dir, current_stage="metadata_and_split")
        write_metadata_and_split(run_dir, epi_cases, ar_cases, int(cfg["split"].get("folds", 5)), seed)
        append_completed_stage(run_dir, "metadata_and_split")
        if bool(cfg["stages"].get("run_r0", True)):
            train_stage("r0", epi_cases, cfg, run_dir, logger)
        if bool(cfg["stages"].get("run_r1", True)):
            train_stage("r1", epi_cases, cfg, run_dir, logger)
        if bool(cfg["training"].get("run_r1_hard", False)):
            train_stage("r1_hard", epi_cases, cfg, run_dir, logger)
        if bool(cfg["stages"].get("eval_aramra", True)):
            evaluate_aramra(ar_cases, cfg, run_dir, logger)
        write_report(run_dir, cfg)
        update_state(run_dir, status="completed", current_stage="completed")
        logger.log("pipeline completed")
    except Exception as exc:
        update_state(run_dir, status="failed", failed_stage=str(exc))
        logger.log(f"pipeline failed: {exc}")
        raise
    finally:
        logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone EpiBios animal-wise OOF + ARAMRA OOD pipeline")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--foreground", action="store_true", help="Run in the current process")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", default=None, help="Explicit run directory")
    parser.add_argument("--resume", default=None, help="Resume/use an existing run directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Missing config: {args.config}")
    cfg = load_config(config_path)
    if not args.foreground and not args.worker:
        launch_background(args, config_path, cfg)
        return
    run_dir = make_run_dir(cfg, args.run_dir, args.resume)
    run_pipeline(cfg, run_dir)


if __name__ == "__main__":
    main()
