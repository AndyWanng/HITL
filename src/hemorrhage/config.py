"""Configuration loading and strongly typed runtime settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProtocolConfig:
    seed: int
    alpha: float
    folds: int
    min_nominal_budget: int
    routine_audit_ratio: dict[str, int]
    tta_modes: list[str]
    uncertainty: dict[str, float]
    selection: dict[str, Any]
    stop: dict[str, float | int]
    review: dict[str, Any]
    loss: dict[str, float]


@dataclass(slots=True)
class ModelConfig:
    model: dict[str, Any]
    preprocessing: dict[str, Any]
    training: dict[str, Any]
    augmentation: dict[str, Any]
    inference: dict[str, Any]
    postprocessing: dict[str, Any]


@dataclass(slots=True)
class RuntimeConfig:
    paths: dict[str, str]
    runtime: dict[str, Any]


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    protocol: ProtocolConfig
    model: ModelConfig
    runtime: RuntimeConfig

    @property
    def data_root(self) -> Path:
        return (self.project_root / self.runtime.paths["data_root"]).resolve()

    @property
    def workspace_root(self) -> Path:
        return (self.project_root / self.runtime.paths["workspace_root"]).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_app_config(
    project_root: Path,
    protocol_path: Path | None = None,
    model_path: Path | None = None,
    runtime_path: Path | None = None,
) -> AppConfig:
    config_root = project_root / "configs"
    protocol_payload = _load_yaml(protocol_path or config_root / "protocol.yaml")
    model_payload = _load_yaml(model_path or config_root / "model.yaml")
    runtime_payload = _load_yaml(runtime_path or config_root / "runtime.local.yaml")
    return AppConfig(
        project_root=project_root.resolve(),
        protocol=ProtocolConfig(**protocol_payload),
        model=ModelConfig(**model_payload),
        runtime=RuntimeConfig(**runtime_payload),
    )
