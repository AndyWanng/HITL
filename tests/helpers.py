"""Test helpers for synthetic datasets."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import yaml


def create_synthetic_case(case_id: str, image_path: Path, label_path: Path, shape: tuple[int, int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    label = np.zeros(shape, dtype=np.int16)
    center = np.array(shape) // 2 + rng.integers(-5, 5, size=3)
    radii = np.maximum(np.array(shape) // 8, 2)
    grid = np.indices(shape).transpose(1, 2, 3, 0)
    dist = (((grid - center) / radii) ** 2).sum(axis=-1)
    blob = dist <= 1.0
    label[blob] = 1
    if seed % 3 == 0:
        label[np.roll(blob, 2, axis=0)] = 3
    if seed % 5 == 0:
        label[np.roll(blob, -2, axis=1)] = 2
    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(image, affine), str(image_path))
    nib.save(nib.Nifti1Image(label, affine), str(label_path))


def create_synthetic_project(root: Path, num_cases: int = 10, shape: tuple[int, int, int] = (160, 122, 80)) -> None:
    (root / "data" / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "data" / "labelsTr").mkdir(parents=True, exist_ok=True)
    (root / "plans").mkdir(parents=True, exist_ok=True)
    for idx in range(num_cases):
        case_id = f"202601{idx:02d}_Study_Rat{2000 + idx}_M_D02_1_1_8_MGE"
        create_synthetic_case(
            case_id,
            root / "data" / "imagesTr" / f"{case_id}_0000.nii.gz",
            root / "data" / "labelsTr" / f"{case_id}.nii.gz",
            shape,
            seed=idx + 1,
        )


def override_project_configs(root: Path, project_root: Path) -> None:
    configs_src = project_root / "configs"
    configs_dst = root / "configs"
    configs_dst.mkdir(parents=True, exist_ok=True)
    for name in ["protocol.yaml", "runtime.local.yaml", "runtime.server.yaml"]:
        configs_dst.joinpath(name).write_text(configs_src.joinpath(name).read_text(encoding="utf-8"), encoding="utf-8")
    model_cfg = yaml.safe_load(configs_src.joinpath("model.yaml").read_text(encoding="utf-8"))
    model_cfg["training"]["batch_size"] = 1
    model_cfg["training"]["patches_per_case"] = 1
    model_cfg["training"]["round0"]["epochs"] = 2
    model_cfg["training"]["finetune"]["epochs"] = 1
    configs_dst.joinpath("model.yaml").write_text(yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8")

