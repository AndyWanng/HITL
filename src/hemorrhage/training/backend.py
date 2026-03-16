"""Residual 3D U-Net training backend."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hemorrhage.config import AppConfig
from hemorrhage.data.nifti import percentile_zscore
from hemorrhage.protocol.metrics import dice_score
from hemorrhage.protocol.rounds import oof_mean_and_variance
from hemorrhage.training.dataset import PreparedCase, RoundPatchDataset
from hemorrhage.training.inference import apply_tta, invert_tta, sliding_window_predict
from hemorrhage.training.losses import protocol_loss
from hemorrhage.training.model import ResidualUNet3D
from hemorrhage.training.postprocess import postprocess_probability_map
from hemorrhage.utils import RunLogger, ensure_dir, write_json_atomic


@dataclass(slots=True)
class TrainingResult:
    checkpoint_path: Path
    loss_history: list[float]
    last_checkpoint_path: Path
    best_epoch: int | None
    best_metric_name: str | None
    best_metric_value: float | None
    val_history: list[dict[str, float]]
    train_case_ids: list[str]
    val_case_ids: list[str]


class Custom3DUNetBackend:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        training_cfg = config.model.training
        self.patch_size = tuple(int(v) for v in training_cfg["patch_size"])
        self.batch_size = int(training_cfg["batch_size"])
        self.num_workers = int(training_cfg.get("num_workers", 0))
        self.patches_per_case = int(training_cfg.get("patches_per_case", 2))
        self.positive_patch_probability = float(training_cfg.get("positive_patch_probability", 0.5))
        self.use_amp = bool(training_cfg.get("amp", True))
        self.augmentation_cfg = dict(config.model.augmentation)
        clip_percentiles = config.model.preprocessing.get("clip_percentiles", (0.5, 99.5))
        self.clip_percentiles = (float(clip_percentiles[0]), float(clip_percentiles[1]))
        self.validation_cfg = dict(training_cfg.get("validation", {}))
        self.validation_enabled = bool(self.validation_cfg.get("enabled", True))
        self.validation_fraction = float(self.validation_cfg.get("fraction", 0.125))
        self.validation_min_cases = int(self.validation_cfg.get("min_cases", 1))
        self.validation_metric = str(self.validation_cfg.get("selection_metric", "macro_dice_postprocessed"))
        self.postprocess_cfg = dict(config.model.postprocessing)

    def build_model(self) -> nn.Module:
        return ResidualUNet3D(
            in_channels=int(self.config.model.model["in_channels"]),
            out_channels=int(self.config.model.model["out_channels"]),
            stage_channels=tuple(int(v) for v in self.config.model.model["stage_channels"]),
            dropout_bottleneck=float(self.config.model.model.get("dropout_bottleneck", 0.0)),
        )

    def resolve_device(self) -> torch.device:
        preferred = str(self.config.runtime.runtime.get("device", "cpu"))
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "cuda" and not self.config.runtime.runtime.get("allow_cpu_fallback", False):
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cpu")

    def prepare_round_data(self, cases: list[dict[str, Any]]) -> list[PreparedCase]:
        prepared: list[PreparedCase] = []
        for case in cases:
            image = self.preprocess_image(case["image"].astype(np.float32))
            prepared.append(
                PreparedCase(
                    case_id=case["case_id"],
                    image=image,
                    target=case["target"].astype(np.float32),
                    voxel_weight=case["voxel_weight"].astype(np.float32),
                    case_weight=float(case["case_weight"]),
                )
            )
        return prepared

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        return percentile_zscore(image.astype(np.float32), *self.clip_percentiles)

    def train_fold(
        self,
        round_index: int,
        fold_id: int,
        train_cases: list[dict[str, Any]],
        checkpoint_dir: Path,
        resume_checkpoint: Path | None = None,
        round0: bool = False,
        logger: RunLogger | None = None,
        training_csv_path: Path | None = None,
        status_json_path: Path | None = None,
    ) -> TrainingResult:
        device = self.resolve_device()
        model = self.build_model().to(device)
        if resume_checkpoint is not None and resume_checkpoint.exists():
            payload = torch.load(resume_checkpoint, map_location="cpu")
            model.load_state_dict(payload["model"])
        train_split, val_split = self._split_train_val_cases(train_cases, fold_id)
        prepared = self.prepare_round_data(train_split)
        dataset = RoundPatchDataset(
            prepared,
            patch_size=self.patch_size,
            patches_per_case=self.patches_per_case,
            positive_patch_probability=self.positive_patch_probability,
            seed=self.config.protocol.seed + round_index * 31 + fold_id,
            augmentation_cfg=self.augmentation_cfg,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=device.type == "cuda",
        )
        phase_cfg = self.config.model.training["round0" if round0 else "finetune"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(phase_cfg["lr"]), weight_decay=float(phase_cfg["weight_decay"]))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and self.use_amp)
        epochs = int(phase_cfg["epochs"])
        loss_history: list[float] = []
        val_history: list[dict[str, float]] = []
        total_steps = len(loader)
        best_metric_value = float("-inf")
        best_epoch: int | None = None

        if training_csv_path is not None:
            ensure_dir(training_csv_path.parent)
            with training_csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "epoch",
                        "mean_loss",
                        "num_steps",
                        "val_macro_dice_raw",
                        "val_macro_dice_postprocessed",
                        "val_micro_dice_raw",
                        "val_micro_dice_postprocessed",
                        "is_new_best",
                        "best_metric_name",
                        "best_metric_value",
                        "best_epoch",
                    ],
                )
                writer.writeheader()

        self._write_status(
            status_json_path,
            {
                "stage": "training",
                "status": "running",
                "round_index": round_index,
                "fold_id": fold_id,
                "device": device.type,
                "epochs_total": epochs,
                "epochs_completed": 0,
                "num_train_cases": len(train_split),
                "num_val_cases": len(val_split),
                "num_steps_per_epoch": total_steps,
                "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None and resume_checkpoint.exists() else None,
                "validation_metric": self.validation_metric if val_split else None,
                "train_case_ids": [case["case_id"] for case in train_split],
                "val_case_ids": [case["case_id"] for case in val_split],
            },
        )
        if logger is not None:
            logger.log(
                f"train start | round={round_index} fold={fold_id} device={device.type} "
                f"train_cases={len(train_split)} val_cases={len(val_split)} epochs={epochs} steps_per_epoch={total_steps} "
                f"resume={'yes' if resume_checkpoint is not None and resume_checkpoint.exists() else 'no'} "
                f"selection_metric={self.validation_metric if val_split else 'last_checkpoint'}"
            )

        model.train()
        for epoch in range(1, epochs + 1):
            running = []
            step_idx = 0
            for step_idx, batch in enumerate(loader, start=1):
                images = batch["image"].to(device)
                targets = batch["target"].to(device)
                voxel_weights = batch["voxel_weight"].to(device)
                case_weights = batch["case_weight"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda" and self.use_amp):
                    logits = model(images)
                    loss = protocol_loss(logits, targets, voxel_weights, case_weights)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running.append(float(loss.detach().cpu()))
            epoch_loss = float(np.mean(running)) if running else 0.0
            loss_history.append(epoch_loss)
            val_metrics = self._evaluate_validation(model, val_split, device) if val_split else {}
            val_history.append(val_metrics)
            current_metric_value = float(val_metrics.get(self.validation_metric, epoch_loss if not val_split else float("-inf")))
            is_new_best = False
            if val_split:
                if current_metric_value > best_metric_value:
                    best_metric_value = current_metric_value
                    best_epoch = epoch
                    is_new_best = True
            elif epoch == epochs:
                best_metric_value = epoch_loss
                best_epoch = epoch
                is_new_best = True
            if training_csv_path is not None:
                with training_csv_path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "epoch",
                            "mean_loss",
                            "num_steps",
                            "val_macro_dice_raw",
                            "val_macro_dice_postprocessed",
                            "val_micro_dice_raw",
                            "val_micro_dice_postprocessed",
                            "is_new_best",
                            "best_metric_name",
                            "best_metric_value",
                            "best_epoch",
                        ],
                    )
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "mean_loss": epoch_loss,
                            "num_steps": total_steps,
                            "val_macro_dice_raw": val_metrics.get("macro_dice_raw"),
                            "val_macro_dice_postprocessed": val_metrics.get("macro_dice_postprocessed"),
                            "val_micro_dice_raw": val_metrics.get("micro_dice_raw"),
                            "val_micro_dice_postprocessed": val_metrics.get("micro_dice_postprocessed"),
                            "is_new_best": is_new_best,
                            "best_metric_name": self.validation_metric if val_split else "last_epoch_loss",
                            "best_metric_value": best_metric_value if best_epoch is not None else None,
                            "best_epoch": best_epoch,
                        }
                    )
            if is_new_best:
                ensure_dir(checkpoint_dir)
                best_checkpoint_path = checkpoint_dir / f"fold_{fold_id}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "round_index": round_index,
                        "fold_id": fold_id,
                        "loss_history": loss_history,
                        "best_epoch": best_epoch,
                        "best_metric_name": self.validation_metric if val_split else "last_epoch_loss",
                        "best_metric_value": best_metric_value,
                        "val_history": val_history,
                        "train_case_ids": [case["case_id"] for case in train_split],
                        "val_case_ids": [case["case_id"] for case in val_split],
                    },
                    best_checkpoint_path,
                )
                if logger is not None:
                    if val_split:
                        logger.log(
                            f"new best validation metric | round={round_index} fold={fold_id} "
                            f"epoch={epoch}/{epochs} {self.validation_metric}={best_metric_value:.6f} checkpoint={best_checkpoint_path}"
                        )
                    else:
                        logger.log(
                            f"new selected checkpoint | round={round_index} fold={fold_id} epoch={epoch}/{epochs} "
                            f"last_epoch_loss={best_metric_value:.6f} checkpoint={best_checkpoint_path}"
                        )
            self._write_status(
                status_json_path,
                {
                    "stage": "training",
                    "status": "running",
                    "round_index": round_index,
                    "fold_id": fold_id,
                    "device": device.type,
                    "epochs_total": epochs,
                    "epochs_completed": epoch,
                    "num_train_cases": len(train_split),
                    "num_val_cases": len(val_split),
                    "num_steps_per_epoch": total_steps,
                    "last_epoch_loss": epoch_loss,
                    "last_step": step_idx if total_steps > 0 else 0,
                    "validation_metric": self.validation_metric if val_split else None,
                    "last_val_metrics": val_metrics,
                    "best_metric_name": self.validation_metric if val_split else "last_epoch_loss",
                    "best_metric_value": best_metric_value if best_epoch is not None else None,
                    "best_epoch": best_epoch,
                    "train_case_ids": [case["case_id"] for case in train_split],
                    "val_case_ids": [case["case_id"] for case in val_split],
                },
            )
            if logger is not None:
                if val_split:
                    logger.log(
                        f"train epoch complete | round={round_index} fold={fold_id} epoch={epoch}/{epochs} "
                        f"mean_loss={epoch_loss:.6f} val_macro_dice_raw={float(val_metrics.get('macro_dice_raw', 0.0)):.6f} "
                        f"val_macro_dice_post={float(val_metrics.get('macro_dice_postprocessed', 0.0)):.6f} "
                        f"best_{self.validation_metric}={best_metric_value if best_epoch is not None else 0.0:.6f}"
                    )
                else:
                    logger.log(f"train epoch complete | round={round_index} fold={fold_id} epoch={epoch}/{epochs} mean_loss={epoch_loss:.6f}")

        ensure_dir(checkpoint_dir)
        checkpoint_path = checkpoint_dir / f"fold_{fold_id}.pt"
        last_checkpoint_path = checkpoint_dir / f"fold_{fold_id}_last.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "round_index": round_index,
                "fold_id": fold_id,
                "loss_history": loss_history,
                "best_epoch": best_epoch,
                "best_metric_name": self.validation_metric if val_split else "last_epoch_loss",
                "best_metric_value": best_metric_value if best_epoch is not None else None,
                "val_history": val_history,
                "train_case_ids": [case["case_id"] for case in train_split],
                "val_case_ids": [case["case_id"] for case in val_split],
            },
            last_checkpoint_path,
        )
        if not checkpoint_path.exists():
            checkpoint_path = last_checkpoint_path
        self._write_status(
            status_json_path,
            {
                "stage": "training",
                "status": "completed",
                "round_index": round_index,
                "fold_id": fold_id,
                "device": device.type,
                "epochs_total": epochs,
                "epochs_completed": epochs,
                "num_train_cases": len(train_split),
                "num_val_cases": len(val_split),
                "num_steps_per_epoch": total_steps,
                "checkpoint_path": str(checkpoint_path),
                "last_checkpoint_path": str(last_checkpoint_path),
                "final_epoch_loss": loss_history[-1] if loss_history else None,
                "validation_metric": self.validation_metric if val_split else None,
                "best_metric_name": self.validation_metric if val_split else "last_epoch_loss",
                "best_metric_value": best_metric_value if best_epoch is not None else None,
                "best_epoch": best_epoch,
                "last_val_metrics": val_history[-1] if val_history else {},
                "train_case_ids": [case["case_id"] for case in train_split],
                "val_case_ids": [case["case_id"] for case in val_split],
            },
        )
        if logger is not None:
            if val_split:
                logger.log(
                    f"train complete | round={round_index} fold={fold_id} "
                    f"selected_checkpoint={checkpoint_path} last_checkpoint={last_checkpoint_path} "
                    f"best_epoch={best_epoch} best_{self.validation_metric}={best_metric_value:.6f}"
                )
            else:
                logger.log(f"train complete | round={round_index} fold={fold_id} checkpoint={checkpoint_path}")
        return TrainingResult(
            checkpoint_path=checkpoint_path,
            loss_history=loss_history,
            last_checkpoint_path=last_checkpoint_path,
            best_epoch=best_epoch,
            best_metric_name=self.validation_metric if val_split else "last_epoch_loss",
            best_metric_value=best_metric_value if best_epoch is not None else None,
            val_history=val_history,
            train_case_ids=[case["case_id"] for case in train_split],
            val_case_ids=[case["case_id"] for case in val_split],
        )

    def predict_oof_fold(
        self,
        checkpoint_path: Path,
        cases: list[dict[str, Any]],
        logger: RunLogger | None = None,
        status_json_path: Path | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        device = self.resolve_device()
        payload = torch.load(checkpoint_path, map_location="cpu")
        model = self.build_model().to(device)
        model.load_state_dict(payload["model"])
        results: dict[str, dict[str, np.ndarray]] = {}
        overlap = float(self.config.model.inference["overlap"])
        patch_size = tuple(int(v) for v in self.config.model.training["patch_size"])
        total_cases = len(cases)
        self._write_status(
            status_json_path,
            {
                "stage": "inference",
                "status": "running",
                "device": device.type,
                "checkpoint_path": str(checkpoint_path),
                "num_cases_total": total_cases,
                "num_cases_completed": 0,
            },
        )
        if logger is not None:
            logger.log(f"inference start | checkpoint={checkpoint_path.name} device={device.type} cases={total_cases}")
        for case_idx, case in enumerate(cases, start=1):
            image = self.preprocess_image(case["image"].astype(np.float32))
            predictions: list[np.ndarray] = []
            for mode in self.config.protocol.tta_modes:
                augmented = apply_tta(image, mode)
                pred = sliding_window_predict(model, augmented, device, patch_size, overlap)
                predictions.append(invert_tta(pred, mode))
            mean_pred, variance = oof_mean_and_variance(predictions)
            results[case["case_id"]] = {"s": mean_pred.astype(np.float32), "q": variance.astype(np.float32)}
            self._write_status(
                status_json_path,
                {
                    "stage": "inference",
                    "status": "running",
                    "device": device.type,
                    "checkpoint_path": str(checkpoint_path),
                    "num_cases_total": total_cases,
                    "num_cases_completed": case_idx,
                    "current_case_id": case["case_id"],
                },
            )
            if logger is not None:
                logger.log(f"inference case complete | checkpoint={checkpoint_path.name} case={case['case_id']} progress={case_idx}/{total_cases}")
        self._write_status(
            status_json_path,
            {
                "stage": "inference",
                "status": "completed",
                "device": device.type,
                "checkpoint_path": str(checkpoint_path),
                "num_cases_total": total_cases,
                "num_cases_completed": total_cases,
            },
        )
        if logger is not None:
            logger.log(f"inference complete | checkpoint={checkpoint_path.name} cases={total_cases}")
        return results

    def predict_external(self, checkpoints: list[Path], image: np.ndarray) -> np.ndarray:
        fold_outputs = []
        for checkpoint in checkpoints:
            fold_results = self.predict_oof_fold(checkpoint, [{"case_id": "external", "image": image}])
            fold_outputs.append(fold_results["external"]["s"])
        return np.mean(np.stack(fold_outputs, axis=0), axis=0).astype(np.float32)

    def _write_status(self, status_json_path: Path | None, payload: dict[str, Any]) -> None:
        if status_json_path is None:
            return
        write_json_atomic(status_json_path, payload)

    def _split_train_val_cases(self, train_cases: list[dict[str, Any]], fold_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.validation_enabled or len(train_cases) < 2:
            return list(train_cases), []

        total_cases = len(train_cases)
        val_count = max(self.validation_min_cases, int(math.ceil(total_cases * self.validation_fraction)))
        val_count = min(val_count, total_cases - 1)
        if val_count <= 0:
            return list(train_cases), []

        rng = np.random.default_rng(self.config.protocol.seed + fold_id * 1009)
        positives = [case for case in sorted(train_cases, key=lambda item: item["case_id"]) if float(np.asarray(case["binary_target"]).sum()) > 0.0]
        negatives = [case for case in sorted(train_cases, key=lambda item: item["case_id"]) if float(np.asarray(case["binary_target"]).sum()) <= 0.0]

        def _pick(bucket: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
            if count <= 0 or not bucket:
                return []
            order = rng.permutation(len(bucket))
            return [bucket[idx] for idx in order[: min(count, len(bucket))]]

        if positives and negatives:
            pos_count = int(round(val_count * len(positives) / total_cases))
            pos_count = min(max(pos_count, 1), len(positives))
            neg_count = min(val_count - pos_count, len(negatives))
            selected = _pick(positives, pos_count) + _pick(negatives, neg_count)
        else:
            selected = _pick(positives or negatives, val_count)

        selected_ids = {case["case_id"] for case in selected}
        remaining = [case for case in sorted(train_cases, key=lambda item: item["case_id"]) if case["case_id"] not in selected_ids]
        while len(selected) < val_count and remaining:
            selected.append(remaining.pop(0))
        val_cases = sorted(selected, key=lambda item: item["case_id"])
        val_ids = {case["case_id"] for case in val_cases}
        train_split = [case for case in sorted(train_cases, key=lambda item: item["case_id"]) if case["case_id"] not in val_ids]
        return train_split, val_cases

    def _evaluate_validation(self, model: nn.Module, val_cases: list[dict[str, Any]], device: torch.device) -> dict[str, float]:
        if not val_cases:
            return {}
        overlap = float(self.config.model.inference["overlap"])
        patch_size = tuple(int(v) for v in self.config.model.training["patch_size"])
        rows = []
        for case in val_cases:
            image = self.preprocess_image(case["image"].astype(np.float32))
            probability = sliding_window_predict(model, image, device, patch_size, overlap)
            target = case["binary_target"].astype(np.uint8)
            pred_raw = (probability >= float(self.postprocess_cfg.get("threshold", self.config.model.inference.get("threshold", 0.5)))).astype(np.uint8)
            pred_post = postprocess_probability_map(
                probability,
                threshold=float(self.postprocess_cfg.get("threshold", self.config.model.inference.get("threshold", 0.5))),
                min_component_voxels=int(self.postprocess_cfg.get("min_component_voxels", 0)),
                largest_only=bool(self.postprocess_cfg.get("keep_largest_component", False)),
            )
            rows.append(
                {
                    "dice_raw": float(dice_score(pred_raw, target)),
                    "dice_postprocessed": float(dice_score(pred_post, target)),
                    "intersection_raw": int(np.logical_and(pred_raw.astype(bool), target.astype(bool)).sum()),
                    "intersection_postprocessed": int(np.logical_and(pred_post.astype(bool), target.astype(bool)).sum()),
                    "pred_positive_voxels_raw": int(pred_raw.sum()),
                    "pred_positive_voxels_postprocessed": int(pred_post.sum()),
                    "gt_positive_voxels": int(target.sum()),
                }
            )
        return self._aggregate_validation_rows(rows)

    def _aggregate_validation_rows(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {}

        def _micro(intersection_key: str, pred_key: str) -> float:
            numerator = 2.0 * sum(float(row[intersection_key]) for row in rows)
            denominator = sum(float(row[pred_key]) for row in rows) + sum(float(row["gt_positive_voxels"]) for row in rows)
            if denominator == 0.0:
                return 1.0
            return float(numerator / denominator)

        return {
            "macro_dice_raw": float(np.mean([float(row["dice_raw"]) for row in rows])),
            "macro_dice_postprocessed": float(np.mean([float(row["dice_postprocessed"]) for row in rows])),
            "micro_dice_raw": _micro("intersection_raw", "pred_positive_voxels_raw"),
            "micro_dice_postprocessed": _micro("intersection_postprocessed", "pred_positive_voxels_postprocessed"),
        }
