"""Generic filesystem, hashing, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def dumps_json(payload: dict[str, Any] | list[Any] | None) -> str:
    if payload is None:
        return "null"
    return json.dumps(payload, sort_keys=True)


def loads_json(raw: str | None, fallback: Any = None) -> Any:
    if raw is None:
        return fallback
    return json.loads(raw)


@dataclass(slots=True)
class RunLogger:
    log_path: Path
    prefix: str
    mirror_stdout: bool = True

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} | {self.prefix} | {message}"
        ensure_dir(self.log_path.parent)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.mirror_stdout:
            print(line, flush=True)

    def child(self, prefix: str) -> RunLogger:
        return RunLogger(log_path=self.log_path, prefix=f"{self.prefix}:{prefix}", mirror_stdout=self.mirror_stdout)


def create_run_logger(log_path: Path, prefix: str, mirror_stdout: bool = True, reset: bool = True) -> RunLogger:
    ensure_dir(log_path.parent)
    if reset:
        log_path.write_text("", encoding="utf-8")
    return RunLogger(log_path=log_path, prefix=prefix, mirror_stdout=mirror_stdout)
