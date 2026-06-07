"""High-level orchestration for the protocol-first hemorrhage pipeline."""

from __future__ import annotations

import csv
import json
from shutil import copy2
from pathlib import Path
from typing import Any

import numpy as np

from hemorrhage.config import AppConfig, load_app_config
from hemorrhage.data.nifti import (
    copy_nifti,
    extract_case_id_from_image,
    inspect_case_geometry,
    label_histogram,
    load_nifti,
    positive_voxels,
    project_binary_mask,
    read_metadata_csv,
    save_nifti,
    scan_case_paths,
    summarize_case_geometries,
    validate_label_codes,
    validate_case_geometry,
)
from hemorrhage.protocol.folds import serpentine_fold_assignment
from hemorrhage.protocol.metrics import dice_score
from hemorrhage.protocol.rounds import (
    build_soft_target,
    build_training_target_payload,
    compute_review_reentry,
    compute_round_summary,
    compute_stop_state,
    compute_uncertainty_from_target,
)
from hemorrhage.protocol.selection import build_audit_pool, compute_case_scores, select_audit, select_routine, split_budget
from hemorrhage.review.io import compute_review_stats, export_review_bundle, import_review_label, normalize_review_metadata, validate_import_dir
from hemorrhage.revision_policy import (
    RevisionCase,
    apply_case_guard,
    apply_component_selector,
    make_revision_case,
    oracle_case_summary,
    predict_adapter,
    train_adapter,
    train_component_selector,
    write_selector_model,
)
from hemorrhage.state.db import StateStore
from hemorrhage.training.backend import Custom3DUNetBackend
from hemorrhage.training.postprocess import postprocess_probability_map
from hemorrhage.utils import RunLogger, create_run_logger, ensure_dir, seed_everything, sha256_file, write_json_atomic


class Pipeline:
    def __init__(self, project_root: Path, runtime_config_path: Path | None = None) -> None:
        self.project_root = project_root
        self.config: AppConfig = load_app_config(project_root, runtime_path=runtime_config_path)
        self.store = StateStore(self.config.workspace_root / "state.db")
        self.store.initialize()
        self.backend = Custom3DUNetBackend(self.config)
        self._ensure_workspace_dirs()
        seed_everything(self.config.protocol.seed)

    def _ensure_workspace_dirs(self) -> None:
        for rel in [
            "artifacts/checkpoints",
            "artifacts/oof",
            "artifacts/labels/raw",
            "artifacts/labels/binary",
            "artifacts/masks",
            "artifacts/soft_targets",
            "artifacts/uncertainty",
            "artifacts/adapters",
            "logs",
            "review",
            "reports",
        ]:
            ensure_dir(self.config.workspace_root / rel)

    def _default_round_progress(self, routine_ids: list[str], audit_ids: list[str]) -> dict[str, bool]:
        has_audit = bool(audit_ids)
        return {
            "routine_imported": not routine_ids,
            "audit_anchor_imported": not has_audit,
            "audit_final_imported": not has_audit,
        }

    def _normalize_round_progress(self, round_record: dict[str, Any]) -> dict[str, bool]:
        defaults = self._default_round_progress(round_record.get("routine_ids", []), round_record.get("audit_ids", []))
        defaults.update({key: bool(value) for key, value in round_record.get("progress", {}).items()})
        return defaults

    def _compute_round_review_status(self, routine_ids: list[str], audit_ids: list[str], progress: dict[str, bool]) -> str:
        if progress.get("routine_imported", False) and progress.get("audit_final_imported", False):
            return "ready_to_finalize"
        if audit_ids and progress.get("audit_anchor_imported", False) and not progress.get("audit_final_imported", False):
            return "awaiting_audit_final"
        return "awaiting_inputs"

    def _selected_case_ids(self, round_record: dict[str, Any]) -> set[str]:
        return set(round_record.get("routine_ids", [])) | set(round_record.get("audit_ids", []))

    def _load_uncertainty_map(self, case: dict[str, Any]) -> np.ndarray:
        if not case.get("current_uncertainty_path"):
            raise RuntimeError(f"Case {case['case_id']} is missing current_uncertainty_path")
        payload = np.load(case["current_uncertainty_path"])
        return payload["uncertainty"].astype(np.float32)

    def _threshold_probability(self, probability: np.ndarray) -> np.ndarray:
        threshold = float(self.config.model.postprocessing.get("threshold", self.config.model.inference.get("threshold", 0.5)))
        return (probability >= threshold).astype(np.uint8)

    def _export_assistance_assets(
        self,
        bundle_root: Path,
        case: dict[str, Any],
        seed_label_path: Path,
        stage: str,
        role: str,
        seed_source: str | None = None,
    ) -> dict[str, str]:
        case_id = case["case_id"]
        image_dst = bundle_root / "images" / f"{case_id}_0000.nii.gz"
        label_dst = bundle_root / "labels_seed" / f"{case_id}.nii.gz"
        mask_dst = bundle_root / "model_mask" / f"{case_id}.nii.gz"
        uncertainty_dst = bundle_root / "uncertainty" / f"{case_id}.nii.gz"
        copy_nifti(Path(case["image_path"]), image_dst)
        copy_nifti(seed_label_path, label_dst)
        reference = load_nifti(Path(case["image_path"]))
        save_nifti(mask_dst, self._threshold_probability(case["previous_oof"]), reference.affine, reference.header, np.uint8)
        save_nifti(uncertainty_dst, self._load_uncertainty_map(case), reference.affine, reference.header, np.float32)
        row = {
            "case_id": case_id,
            "role": role,
            "stage": stage,
            "image_path": str(image_dst),
            "seed_label_path": str(label_dst),
            "model_mask_path": str(mask_dst),
            "uncertainty_path": str(uncertainty_dst),
        }
        if seed_source is not None:
            row["seed_source"] = seed_source
        return row

    def _export_seed_only_assets(
        self,
        bundle_root: Path,
        case: dict[str, Any],
        seed_label_path: Path,
        stage: str,
        role: str,
    ) -> dict[str, str]:
        case_id = case["case_id"]
        image_dst = bundle_root / "images" / f"{case_id}_0000.nii.gz"
        label_dst = bundle_root / "labels_seed" / f"{case_id}.nii.gz"
        copy_nifti(Path(case["image_path"]), image_dst)
        copy_nifti(seed_label_path, label_dst)
        return {
            "case_id": case_id,
            "role": role,
            "stage": stage,
            "image_path": str(image_dst),
            "seed_label_path": str(label_dst),
        }

    def _export_audit_final_bundle(self, round_index: int, audit_ids: list[str], case_map: dict[str, dict[str, Any]]) -> Path:
        audit_final_root = ensure_dir(self.config.workspace_root / "review" / f"round_{round_index}" / "audit_final")
        rows = []
        for case_id in audit_ids:
            stats_row = self.store.get_review_stats(round_index, case_id)
            if stats_row is not None and stats_row.get("audit_anchor_label_path"):
                seed_path = Path(stats_row["audit_anchor_label_path"])
                seed_source = "anchor_label"
            else:
                seed_path = Path(case_map[case_id]["current_raw_label_path"])
                seed_source = "current_label_pending_anchor_update"
            rows.append(
                self._export_assistance_assets(
                    audit_final_root,
                    case_map[case_id],
                    seed_path,
                    stage="audit_final",
                    role="audit",
                    seed_source=seed_source,
                )
            )
        export_review_bundle(audit_final_root, rows)
        return audit_final_root

    def _merge_review_stats(self, round_index: int, case_id: str, role: str, updates: dict[str, Any]) -> dict[str, Any]:
        record = self.store.get_review_stats(round_index, case_id) or {
            "round_index": round_index,
            "case_id": case_id,
            "role": role,
            "routine_final_label_path": None,
            "audit_anchor_label_path": None,
            "audit_final_label_path": None,
            "edit_ratio": None,
            "whole_volume_edit_ratio": None,
            "modified_slices_count": None,
            "anchor_assisted_dice": None,
            "review_time": None,
            "anchor_time": None,
            "assisted_time": None,
            "warnings": [],
        }
        record["role"] = role
        warnings = list(record.get("warnings", []))
        for warning in updates.pop("warnings", []):
            if warning not in warnings:
                warnings.append(warning)
        record.update(updates)
        record["warnings"] = warnings
        self.store.upsert_review_stats(round_index, case_id, record)
        return record

    def _read_review_metadata(self, input_dir: Path) -> dict[str, dict[str, str]]:
        metadata_csv = input_dir / "metadata.csv"
        if not metadata_csv.exists():
            raise FileNotFoundError(metadata_csv)
        return {row["case_id"]: row for row in read_metadata_csv(metadata_csv)}

    def _parse_optional_time(self, metadata: dict[str, str], field: str, warnings: list[str], case_id: str) -> float | None:
        raw = metadata.get(field, "").strip()
        if raw == "":
            warnings.append(f"missing_{field}")
            return None
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"Case {case_id} has invalid {field}: {raw}") from exc

    def init_project(self) -> None:
        logger = self._create_run_logger("init-project")
        logger.log(f"init start | project_root={self.project_root} data_root={self.config.data_root} workspace_root={self.config.workspace_root}")
        self.store.initialize()
        cases = scan_case_paths(self.config.data_root)
        logger.log(f"scan complete | num_cases={len(cases)}")
        fold_items: list[tuple[str, int]] = []
        loaded: dict[str, dict[str, Any]] = {}
        geometry_cases = []
        round_raw_dir = ensure_dir(self.config.workspace_root / "artifacts" / "labels" / "raw" / "round_0")
        round_bin_dir = ensure_dir(self.config.workspace_root / "artifacts" / "labels" / "binary" / "round_0")
        init_report_dir = ensure_dir(self.config.workspace_root / "reports" / "init")
        preprocessing_cfg = self.config.model.preprocessing
        spacing_tolerance = float(preprocessing_cfg.get("spacing_tolerance", 1.0e-4))
        affine_tolerance = float(preprocessing_cfg.get("affine_tolerance", 1.0e-4))
        enforce_geometry_match = bool(preprocessing_cfg.get("enforce_geometry_match", True))
        for case in cases:
            geometry = inspect_case_geometry(
                case.case_id,
                case.image_path,
                case.label_path,
                spacing_tolerance=spacing_tolerance,
                affine_tolerance=affine_tolerance,
            )
            geometry_cases.append(geometry)
            if enforce_geometry_match:
                validate_case_geometry(geometry)
            raw = load_nifti(case.label_path)
            raw_codes = raw.data.astype(np.int16)
            validate_label_codes(raw_codes)
            binary = project_binary_mask(raw_codes)
            v0 = positive_voxels(binary)
            fold_items.append((case.case_id, v0))
            raw_dst = round_raw_dir / f"{case.case_id}.nii.gz"
            bin_dst = round_bin_dir / f"{case.case_id}.nii.gz"
            copy_nifti(case.label_path, raw_dst)
            save_nifti(bin_dst, binary, raw.affine, raw.header, np.uint8)
            loaded[case.case_id] = {
                "raw_path": str(raw_dst),
                "binary_path": str(bin_dst),
                "histogram": label_histogram(raw_codes),
                "v0": v0,
                "image_path": str(case.image_path),
                "source_label_path": str(case.label_path),
                "subject_id": case.subject_id,
            }
        fold_map = serpentine_fold_assignment(fold_items, self.config.protocol.folds)
        logger.log(f"fold assignment complete | folds={self.config.protocol.folds}")
        for case_id, payload in loaded.items():
            self.store.upsert_case(
                {
                    "case_id": case_id,
                    "image_path": payload["image_path"],
                    "source_label_path": payload["source_label_path"],
                    "subject_id": payload["subject_id"],
                    "fold_id": fold_map[case_id],
                    "v0": payload["v0"],
                    "review_count": 0,
                    "last_review_round": -1,
                    "earliest_eligible_round": 1,
                    "current_raw_label_path": payload["raw_path"],
                    "current_binary_label_path": payload["binary_path"],
                    "current_oof_path": None,
                    "current_soft_target_path": None,
                    "current_uncertainty_path": None,
                    "metadata": {"label_histogram": payload["histogram"]},
                }
            )
            self.store.add_artifact("raw_label", payload["raw_path"], sha256_file(Path(payload["raw_path"])), round_index=0, case_id=case_id, metadata={"histogram": payload["histogram"]})
            self.store.add_artifact("binary_label", payload["binary_path"], sha256_file(Path(payload["binary_path"])), round_index=0, case_id=case_id)
        self._write_init_audit_report(init_report_dir, geometry_cases)
        self.store.upsert_round(0, {"status": "initialized", "budget": None, "metrics": {}, "stop_state": {}})
        write_json_atomic(
            self.config.workspace_root / "project_snapshot.json",
            {
                "project_root": str(self.project_root),
                "data_root": str(self.config.data_root),
                "workspace_root": str(self.config.workspace_root),
                "seed": self.config.protocol.seed,
            },
        )
        logger.log("init complete")

    def _load_case_payload(self, case: dict[str, Any]) -> dict[str, Any]:
        image = load_nifti(Path(case["image_path"])).data.astype(np.float32)
        raw_codes = load_nifti(Path(case["current_raw_label_path"])).data.astype(np.int16)
        binary = load_nifti(Path(case["current_binary_label_path"])).data.astype(np.uint8)
        payload = {
            **case,
            "image": image,
            "raw_codes": raw_codes,
            "current_binary_label": binary,
            "previous_binary_label": binary.astype(np.float32),
        }
        if case.get("current_oof_path"):
            oof_payload = np.load(case["current_oof_path"])
            payload["previous_oof"] = oof_payload["s"].astype(np.float32)
            payload["previous_q"] = oof_payload["q"].astype(np.float32)
        else:
            zeros = np.zeros_like(binary, dtype=np.float32)
            payload["previous_oof"] = zeros
            payload["previous_q"] = zeros
        return payload

    def train_round0(self) -> None:
        logger = self._create_run_logger("train-round0", round_index=0)
        if self.store.get_round(0) is None:
            raise RuntimeError("Project not initialized. Run init-project first.")
        cases = [self._load_case_payload(case) for case in self.store.list_cases()]
        logger.log(f"round0 start | cases={len(cases)} folds={self.config.protocol.folds} device={self.config.runtime.runtime.get('device', 'cpu')}")
        checkpoint_dir = ensure_dir(self.config.workspace_root / "artifacts" / "checkpoints" / "round_0")
        oof_dir = ensure_dir(self.config.workspace_root / "artifacts" / "oof" / "round_0")
        mask_dir = ensure_dir(self.config.workspace_root / "artifacts" / "masks" / "round_0")
        soft_dir = ensure_dir(self.config.workspace_root / "artifacts" / "soft_targets" / "round_0")
        unc_dir = ensure_dir(self.config.workspace_root / "artifacts" / "uncertainty" / "round_0")
        report_dir = ensure_dir(self.config.workspace_root / "reports" / "round_0")
        checkpoints: dict[int, Path] = {}
        fold_case_ids: dict[int, list[str]] = {}

        for fold_id in range(1, self.config.protocol.folds + 1):
            train_cases = []
            holdout_cases = []
            for case in cases:
                if int(case["fold_id"]) == fold_id:
                    holdout_cases.append({"case_id": case["case_id"], "image": case["image"]})
                else:
                    train_cases.append(
                        {
                            "case_id": case["case_id"],
                            "image": case["image"],
                            "target": case["current_binary_label"].astype(np.float32),
                            "binary_target": case["current_binary_label"].astype(np.uint8),
                            "voxel_weight": np.ones_like(case["current_binary_label"], dtype=np.float32),
                            "case_weight": 1.0,
                        }
                    )
            fold_case_ids[fold_id] = [item["case_id"] for item in holdout_cases]
            fold_logger = logger.child(f"fold={fold_id}")
            fold_logger.log(f"fold start | train_cases={len(train_cases)} holdout_cases={len(holdout_cases)}")
            result = self.backend.train_fold(
                round_index=0,
                fold_id=fold_id,
                train_cases=train_cases,
                checkpoint_dir=checkpoint_dir,
                round0=True,
                logger=fold_logger,
                training_csv_path=report_dir / f"fold_{fold_id}_train.csv",
                status_json_path=report_dir / f"fold_{fold_id}_train_status.json",
            )
            checkpoints[fold_id] = result.checkpoint_path
            self.store.add_artifact(
                "checkpoint",
                str(result.checkpoint_path),
                sha256_file(result.checkpoint_path),
                round_index=0,
                metadata={
                    "fold_id": fold_id,
                    "loss_history": result.loss_history,
                    "best_epoch": result.best_epoch,
                    "best_metric_name": result.best_metric_name,
                    "best_metric_value": result.best_metric_value,
                    "train_case_ids": result.train_case_ids,
                    "val_case_ids": result.val_case_ids,
                },
            )
            self.store.add_artifact(
                "checkpoint_last",
                str(result.last_checkpoint_path),
                sha256_file(result.last_checkpoint_path),
                round_index=0,
                metadata={"fold_id": fold_id, "loss_history": result.loss_history},
            )
            self._write_fold_training_log(report_dir, fold_id, result.loss_history)
            predictions = self.backend.predict_oof_fold(
                result.checkpoint_path,
                holdout_cases,
                logger=fold_logger,
                status_json_path=report_dir / f"fold_{fold_id}_inference_status.json",
            )
            self._write_fold_inference_log(report_dir, fold_id, list(predictions))
            for case_id, pred in predictions.items():
                oof_path = oof_dir / f"{case_id}.npz"
                np.savez_compressed(oof_path, s=pred["s"], q=pred["q"])
                self.store.add_artifact("oof_prediction", str(oof_path), sha256_file(oof_path), round_index=0, case_id=case_id)
            fold_logger.log(f"fold complete | checkpoint={result.checkpoint_path} oof_cases={len(predictions)}")

        for case in cases:
            case_id = case["case_id"]
            oof_payload = np.load(oof_dir / f"{case_id}.npz")
            soft_target = build_soft_target(case["current_binary_label"], oof_payload["s"], self.config.protocol.alpha)
            uncertainty = compute_uncertainty_from_target(soft_target, self.config.protocol.alpha)
            soft_path = soft_dir / f"{case_id}.npz"
            unc_path = unc_dir / f"{case_id}.npz"
            np.savez_compressed(soft_path, target=soft_target)
            np.savez_compressed(unc_path, uncertainty=uncertainty)
            self._save_postprocessed_mask(round_index=0, case_id=case_id, probability=oof_payload["s"], reference_path=Path(case["current_binary_label_path"]), output_dir=mask_dir)
            stored = self.store.get_case(case_id)
            stored["current_oof_path"] = str(oof_dir / f"{case_id}.npz")
            stored["current_soft_target_path"] = str(soft_path)
            stored["current_uncertainty_path"] = str(unc_path)
            self.store.upsert_case(stored)
            self.store.add_artifact("soft_target", str(soft_path), sha256_file(soft_path), round_index=0, case_id=case_id)
            self.store.add_artifact("uncertainty", str(unc_path), sha256_file(unc_path), round_index=0, case_id=case_id)
        latest_cases = [self._load_case_payload(case) for case in self.store.list_cases()]
        oof_metrics = self._compute_oof_metrics(latest_cases)
        self._write_oof_reports(report_dir, oof_metrics)
        self._rewrite_fold_inference_logs(report_dir, fold_case_ids, oof_metrics["fold_rows"])
        self.store.upsert_round(0, {"status": "completed", "budget": None, "metrics": {"oof": oof_metrics["summary"]}, "stop_state": {}})
        logger.log(f"round0 complete | macro_dice_raw={oof_metrics['summary']['macro_dice_raw']:.6f} macro_dice_post={oof_metrics['summary']['macro_dice_postprocessed']:.6f}")

    def plan_round(self, round_index: int, budget: int) -> None:
        logger = self._create_run_logger("plan-round", round_index=round_index)
        previous_round = self.store.get_round(round_index - 1)
        if previous_round is None or previous_round["status"] != "completed":
            raise RuntimeError(f"Previous round {round_index - 1} must be completed before planning round {round_index}")
        cases = [self._load_case_payload(case) for case in self.store.list_cases()]
        eligible = [case for case in cases if round_index >= int(case["earliest_eligible_round"])]
        actual_budget = min(budget, len(eligible))
        logger.log(f"plan start | round={round_index} requested_budget={budget} eligible_cases={len(eligible)} actual_budget={actual_budget}")
        if actual_budget == 0:
            self.store.upsert_round(round_index, {"status": "empty", "budget": budget, "metrics": {}, "stop_state": {}})
            logger.log("plan complete | empty round")
            return
        routine_budget, audit_budget = split_budget(actual_budget)
        scored = compute_case_scores(eligible, self.config.protocol)
        routine = select_routine(scored, routine_budget)
        audit_pool = build_audit_pool(scored, {item.case_id for item in routine}, audit_budget)
        audit = select_audit(audit_pool, audit_budget, self.config.protocol.folds)
        routine_ids = [item.case_id for item in routine]
        audit_ids = [item.case_id for item in audit]
        self.store.replace_case_metrics(
            round_index,
            [
                {
                    "case_id": item.case_id,
                    "role": item.role,
                    "d": item.d,
                    "u": item.u,
                    "c": item.c,
                    "d_bar": item.d_bar,
                    "u_bar": item.u_bar,
                    "c_bar": item.c_bar,
                    "benefit": item.benefit,
                    "score": item.score,
                    "score_eff": item.score_eff,
                }
                for item in scored
            ],
        )
        self.store.upsert_round(
            round_index,
            {
                "status": self._compute_round_review_status(routine_ids, audit_ids, self._default_round_progress(routine_ids, audit_ids)),
                "budget": budget,
                "routine_ids": routine_ids,
                "audit_ids": audit_ids,
                "metrics": {},
                "stop_state": {},
                "progress": self._default_round_progress(routine_ids, audit_ids),
            },
        )
        routine_root = ensure_dir(self.config.workspace_root / "review" / f"round_{round_index}" / "routine")
        audit_anchor_root = ensure_dir(self.config.workspace_root / "review" / f"round_{round_index}" / "audit_anchor")
        case_map = {case["case_id"]: case for case in cases}
        routine_rows = [
            self._export_assistance_assets(
                routine_root,
                case_map[case_id],
                Path(case_map[case_id]["current_raw_label_path"]),
                stage="routine",
                role="routine",
            )
            for case_id in routine_ids
        ]
        audit_rows = [
            self._export_seed_only_assets(
                audit_anchor_root,
                case_map[case_id],
                Path(case_map[case_id]["current_raw_label_path"]),
                stage="audit_anchor",
                role="audit",
            )
            for case_id in audit_ids
        ]
        export_review_bundle(routine_root, routine_rows)
        export_review_bundle(audit_anchor_root, audit_rows)
        audit_final_root = self._export_audit_final_bundle(round_index, audit_ids, case_map)
        logger.log(
            f"plan complete | routine={len(routine_ids)} audit={len(audit_ids)} "
            f"routine_bundle={routine_root} audit_anchor_bundle={audit_anchor_root} "
            f"audit_final_bundle={audit_final_root}"
        )

    def import_routine(self, round_index: int, input_dir: Path) -> None:
        logger = self._create_run_logger("import-routine", round_index=round_index)
        round_record = self.store.get_round(round_index)
        if round_record is None or round_record["status"] not in {"awaiting_inputs", "awaiting_audit_final", "ready_to_finalize"}:
            raise RuntimeError(f"Round {round_index} is not awaiting routine import")
        progress = self._normalize_round_progress(round_record)
        if not round_record["routine_ids"]:
            logger.log(f"routine import skipped | round={round_index} routine_cases=0")
            return
        if progress["routine_imported"]:
            raise RuntimeError(f"Routine labels for round {round_index} have already been imported")
        logger.log(f"routine import start | round={round_index} input_dir={input_dir}")
        labels_dir = input_dir / "labels"
        if not labels_dir.exists():
            raise FileNotFoundError(labels_dir)
        metadata_rows = self._read_review_metadata(input_dir)
        output_dir = ensure_dir(self.config.workspace_root / "artifacts" / "labels" / "raw" / f"round_{round_index}" / "routine")
        for case_id in round_record["routine_ids"]:
            label_path = labels_dir / f"{case_id}.nii.gz"
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            case = self._load_case_payload(self.store.get_case(case_id))
            metadata = metadata_rows.get(case_id, {"case_id": case_id})
            warnings: list[str] = []
            review_time = self._parse_optional_time(metadata, "review_time", warnings, case_id)
            imported = import_review_label(label_path)
            artifact_path = output_dir / f"{case_id}.nii.gz"
            save_nifti(artifact_path, imported["raw"], imported["affine"], imported["header"], np.int16)
            stats = compute_review_stats(case["previous_binary_label"], imported["binary"])
            self.store.upsert_review(round_index, case_id, "routine", "routine", str(artifact_path), metadata)
            self._merge_review_stats(
                round_index,
                case_id,
                "routine",
                {
                    "routine_final_label_path": str(artifact_path),
                    "edit_ratio": stats["edit_ratio"],
                    "whole_volume_edit_ratio": stats["whole_volume_edit_ratio"],
                    "modified_slices_count": stats["modified_slices_count"],
                    "review_time": review_time,
                    "warnings": warnings,
                },
            )
        progress["routine_imported"] = True
        updated_round = {
            **round_record,
            "status": self._compute_round_review_status(round_record["routine_ids"], round_record["audit_ids"], progress),
            "progress": progress,
        }
        self.store.upsert_round(round_index, updated_round)
        logger.log(f"routine import complete | routine_cases={len(round_record['routine_ids'])}")

    def import_audit_anchor(self, round_index: int, input_dir: Path) -> None:
        logger = self._create_run_logger("import-audit-anchor", round_index=round_index)
        round_record = self.store.get_round(round_index)
        if round_record is None or round_record["status"] not in {"awaiting_inputs", "awaiting_audit_final", "ready_to_finalize"}:
            raise RuntimeError(f"Round {round_index} is not awaiting audit-anchor import")
        progress = self._normalize_round_progress(round_record)
        if not round_record["audit_ids"]:
            logger.log(f"audit anchor import skipped | round={round_index} audit_cases=0")
            return
        if progress["audit_anchor_imported"]:
            raise RuntimeError(f"Audit anchor labels for round {round_index} have already been imported")
        logger.log(f"audit anchor import start | round={round_index} input_dir={input_dir}")
        labels_dir = input_dir / "labels"
        if not labels_dir.exists():
            raise FileNotFoundError(labels_dir)
        metadata_rows = self._read_review_metadata(input_dir)
        output_dir = ensure_dir(self.config.workspace_root / "artifacts" / "labels" / "raw" / f"round_{round_index}" / "audit_anchor")
        case_map = {case["case_id"]: case for case in [self._load_case_payload(case) for case in self.store.list_cases()]}
        for case_id in round_record["audit_ids"]:
            label_path = labels_dir / f"{case_id}.nii.gz"
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            metadata = metadata_rows.get(case_id, {"case_id": case_id})
            warnings: list[str] = []
            anchor_time = self._parse_optional_time(metadata, "anchor_time", warnings, case_id)
            imported = import_review_label(label_path)
            artifact_path = output_dir / f"{case_id}.nii.gz"
            save_nifti(artifact_path, imported["raw"], imported["affine"], imported["header"], np.int16)
            self.store.upsert_review(round_index, case_id, "audit", "audit_anchor", str(artifact_path), metadata)
            self._merge_review_stats(
                round_index,
                case_id,
                "audit",
                {
                    "audit_anchor_label_path": str(artifact_path),
                    "anchor_time": anchor_time,
                    "warnings": warnings,
                },
            )

        audit_final_root = self._export_audit_final_bundle(round_index, round_record["audit_ids"], case_map)
        progress["audit_anchor_imported"] = True
        updated_round = {
            **round_record,
            "status": self._compute_round_review_status(round_record["routine_ids"], round_record["audit_ids"], progress),
            "progress": progress,
        }
        self.store.upsert_round(round_index, updated_round)
        logger.log(
            f"audit anchor import complete | audit_cases={len(round_record['audit_ids'])} "
            f"audit_final_bundle_refreshed={audit_final_root}"
        )

    def import_audit_final(self, round_index: int, input_dir: Path) -> None:
        logger = self._create_run_logger("import-audit-final", round_index=round_index)
        round_record = self.store.get_round(round_index)
        if round_record is None or round_record["status"] not in {"awaiting_audit_final", "awaiting_inputs", "ready_to_finalize"}:
            raise RuntimeError(f"Round {round_index} is not awaiting audit-final import")
        progress = self._normalize_round_progress(round_record)
        if not round_record["audit_ids"]:
            logger.log(f"audit final import skipped | round={round_index} audit_cases=0")
            return
        if not progress["audit_anchor_imported"]:
            raise RuntimeError(f"Round {round_index} is missing audit anchor labels")
        if progress["audit_final_imported"]:
            raise RuntimeError(f"Audit final labels for round {round_index} have already been imported")
        logger.log(f"audit final import start | round={round_index} input_dir={input_dir}")
        labels_dir = input_dir / "labels"
        if not labels_dir.exists():
            raise FileNotFoundError(labels_dir)
        metadata_rows = self._read_review_metadata(input_dir)
        output_dir = ensure_dir(self.config.workspace_root / "artifacts" / "labels" / "raw" / f"round_{round_index}" / "audit_final")
        for case_id in round_record["audit_ids"]:
            label_path = labels_dir / f"{case_id}.nii.gz"
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            case = self._load_case_payload(self.store.get_case(case_id))
            metadata = metadata_rows.get(case_id, {"case_id": case_id})
            warnings: list[str] = []
            assisted_time = self._parse_optional_time(metadata, "assisted_time", warnings, case_id)
            imported = import_review_label(label_path)
            artifact_path = output_dir / f"{case_id}.nii.gz"
            save_nifti(artifact_path, imported["raw"], imported["affine"], imported["header"], np.int16)
            self.store.upsert_review(round_index, case_id, "audit", "audit_final", str(artifact_path), metadata)
            stats_row = self.store.get_review_stats(round_index, case_id)
            if stats_row is None or not stats_row.get("audit_anchor_label_path"):
                raise RuntimeError(f"Missing audit anchor stats for {case_id}")
            anchor = import_review_label(Path(stats_row["audit_anchor_label_path"]))
            stats = compute_review_stats(case["previous_binary_label"], imported["binary"], anchor["binary"])
            self._merge_review_stats(
                round_index,
                case_id,
                "audit",
                {
                    "audit_final_label_path": str(artifact_path),
                    "edit_ratio": stats["edit_ratio"],
                    "whole_volume_edit_ratio": stats["whole_volume_edit_ratio"],
                    "modified_slices_count": stats["modified_slices_count"],
                    "anchor_assisted_dice": stats["anchor_assisted_dice"],
                    "assisted_time": assisted_time,
                    "warnings": warnings,
                },
            )
        progress["audit_final_imported"] = True
        updated_round = {
            **round_record,
            "status": self._compute_round_review_status(round_record["routine_ids"], round_record["audit_ids"], progress),
            "progress": progress,
        }
        self.store.upsert_round(round_index, updated_round)
        logger.log(f"audit final import complete | audit_cases={len(round_record['audit_ids'])}")

    def import_phase1(self, round_index: int, input_dir: Path) -> None:
        logger = self._create_run_logger("import-phase1", round_index=round_index)
        logger.log("deprecated command | mapping import-phase1 to import-routine + import-audit-anchor")
        self.import_routine(round_index, input_dir)
        self.import_audit_anchor(round_index, input_dir)

    def import_phase2(self, round_index: int, input_dir: Path) -> None:
        logger = self._create_run_logger("import-phase2", round_index=round_index)
        logger.log("deprecated command | mapping import-phase2 to import-audit-final")
        self.import_audit_final(round_index, input_dir)

    def finalize_round(self, round_index: int) -> None:
        logger = self._create_run_logger("finalize-round", round_index=round_index)
        round_record = self.store.get_round(round_index)
        if round_record is None or round_record["status"] not in {"ready_to_finalize", "empty"}:
            raise RuntimeError(f"Round {round_index} is not ready to finalize")
        if round_record["status"] == "empty":
            logger.log(f"finalize skipped | round={round_index} status=empty")
            return
        progress = self._normalize_round_progress(round_record)
        if not progress["routine_imported"] or not progress["audit_final_imported"]:
            raise RuntimeError(f"Round {round_index} is missing required review imports")
        revision_cfg = dict(self.config.model.training.get("revision_policy", {}))
        if bool(revision_cfg.get("enabled", False)):
            self._finalize_round_revision_policy(round_index, logger, round_record, revision_cfg)
            return
        previous_cases = [self._load_case_payload(case) for case in self.store.list_cases()]
        logger.log(f"finalize start | round={round_index} cases={len(previous_cases)} device={self.config.runtime.runtime.get('device', 'cpu')}")
        review_stats_map = {row["case_id"]: row for row in self.store.list_review_stats(round_index)}
        raw_dir = ensure_dir(self.config.workspace_root / "artifacts" / "labels" / "raw" / f"round_{round_index}")
        bin_dir = ensure_dir(self.config.workspace_root / "artifacts" / "labels" / "binary" / f"round_{round_index}")
        oof_dir = ensure_dir(self.config.workspace_root / "artifacts" / "oof" / f"round_{round_index}")
        mask_dir = ensure_dir(self.config.workspace_root / "artifacts" / "masks" / f"round_{round_index}")
        soft_dir = ensure_dir(self.config.workspace_root / "artifacts" / "soft_targets" / f"round_{round_index}")
        unc_dir = ensure_dir(self.config.workspace_root / "artifacts" / "uncertainty" / f"round_{round_index}")
        review_records: dict[str, dict[str, Any]] = {}
        prepared_cases: list[dict[str, Any]] = []
        previous_targets: dict[str, np.ndarray] = {}
        current_targets: dict[str, np.ndarray] = {}
        previous_predictions: dict[str, np.ndarray] = {}
        previous_binary_labels: dict[str, np.ndarray] = {}
        alignment_cfg = dict(self.config.model.training.get("alignment", {}))
        alignment_teacher_oof_dir, alignment_teacher_source = self._resolve_alignment_teacher_oof_dir(round_index, alignment_cfg)
        alignment_teacher_checkpoint_dir, alignment_checkpoint_source = self._resolve_alignment_teacher_checkpoint_dir(round_index, alignment_cfg)
        if bool(alignment_cfg.get("enabled", False)):
            logger.log(
                f"alignment enabled | teacher_oof={alignment_teacher_source} "
                f"teacher_checkpoint={alignment_checkpoint_source}"
            )

        for case in previous_cases:
            case_id = case["case_id"]
            previous_targets[case_id] = np.load(case["current_soft_target_path"])["target"].astype(np.float32)
            previous_predictions[case_id] = case["previous_oof"].astype(np.float32)
            previous_binary = case["current_binary_label"].astype(np.uint8)
            previous_binary_labels[case_id] = previous_binary
            current_raw = load_nifti(Path(case["current_raw_label_path"]))
            current_binary = previous_binary.copy()
            role = None
            if case_id in round_record["routine_ids"]:
                role = "routine"
                stats_row = review_stats_map.get(case_id)
                if stats_row is None or not stats_row.get("routine_final_label_path"):
                    raise RuntimeError(f"Missing routine review stats for {case_id}")
                imported = import_review_label(Path(stats_row["routine_final_label_path"]))
                current_raw = load_nifti(Path(stats_row["routine_final_label_path"]))
                current_binary = imported["binary"]
                review_records[case_id] = {**stats_row, "previous_binary": previous_binary, "final_binary": current_binary}
            elif case_id in round_record["audit_ids"]:
                role = "audit"
                stats_row = review_stats_map.get(case_id)
                if stats_row is None or not stats_row.get("audit_anchor_label_path") or not stats_row.get("audit_final_label_path"):
                    raise RuntimeError(f"Missing audit review stats for {case_id}")
                anchor = import_review_label(Path(stats_row["audit_anchor_label_path"]))
                final = import_review_label(Path(stats_row["audit_final_label_path"]))
                current_raw = load_nifti(Path(stats_row["audit_final_label_path"]))
                current_binary = final["binary"]
                review_records[case_id] = {
                    **stats_row,
                    "previous_binary": previous_binary,
                    "anchor_binary": anchor["binary"],
                    "final_binary": current_binary,
                }
            raw_path = raw_dir / f"{case_id}.nii.gz"
            bin_path = bin_dir / f"{case_id}.nii.gz"
            copy_nifti(Path(current_raw.header.get_filename() or current_raw.header["descrip"].tobytes().decode(errors="ignore")), raw_path) if False else save_nifti(raw_path, current_raw.data.astype(np.int16), current_raw.affine, current_raw.header, np.int16)
            save_nifti(bin_path, current_binary, current_raw.affine, current_raw.header, np.uint8)
            updated = self.store.get_case(case_id)
            if role is not None:
                updated["review_count"] = int(updated["review_count"]) + 1
                updated["last_review_round"] = round_index
                anchor_dice = float(review_records[case_id].get("anchor_assisted_dice") or 1.0)
                updated["earliest_eligible_round"] = compute_review_reentry(anchor_dice, self.config.protocol.review, round_index, role)
            updated["current_raw_label_path"] = str(raw_path)
            updated["current_binary_label_path"] = str(bin_path)
            self.store.upsert_case(updated)
            training_payload = build_training_target_payload(
                previous_binary_mask=previous_binary,
                current_binary_mask=current_binary,
                oof_probability=case["previous_oof"],
                alpha=self.config.protocol.alpha,
                reviewed=role is not None,
                floor_weight=float(self.config.protocol.loss["voxel_weight_floor"]),
                alignment_cfg=alignment_cfg,
                teacher_probability=self._load_alignment_teacher_probability(
                    case_id,
                    alignment_teacher_oof_dir,
                    case["previous_oof"],
                    alignment_cfg,
                ),
            )
            soft_target = training_payload["target"]
            current_targets[case_id] = soft_target
            prepared_cases.append(
                {
                    "case_id": case_id,
                    "fold_id": int(case["fold_id"]),
                    "image": case["image"],
                    "target": soft_target,
                    "binary_target": current_binary.astype(np.uint8),
                    "voxel_weight": training_payload["voxel_weight"],
                    "case_weight": float(self.config.protocol.loss["case_weight_reviewed"] if role is not None else self.config.protocol.loss["case_weight_unreviewed"]),
                    "alignment_core_mask": training_payload["alignment_core_mask"],
                    "alignment_core_target": training_payload["alignment_core_target"],
                    "alignment_core_weight": training_payload["alignment_core_weight"],
                    "alignment_sampling_mask": training_payload["alignment_sampling_mask"],
                    "alignment_teacher": training_payload["alignment_teacher"],
                    "alignment_trust_weight": training_payload["alignment_trust_weight"],
                }
            )

        checkpoint_dir = ensure_dir(self.config.workspace_root / "artifacts" / "checkpoints" / f"round_{round_index}")
        report_dir = ensure_dir(self.config.workspace_root / "reports" / f"round_{round_index}")
        round_metrics_history = []
        fold_case_ids: dict[int, list[str]] = {}
        fold_selection_records: list[dict[str, Any]] = []
        for existing_round in range(1, round_index):
            existing = self.store.get_round(existing_round)
            if existing is not None and existing["metrics"]:
                round_metrics_history.append(existing["metrics"])

        for fold_id in range(1, self.config.protocol.folds + 1):
            train_cases = [case for case in prepared_cases if case["fold_id"] != fold_id]
            holdout_cases = [{"case_id": case["case_id"], "image": case["image"]} for case in prepared_cases if case["fold_id"] == fold_id]
            fold_case_ids[fold_id] = [item["case_id"] for item in holdout_cases]
            previous_ckpt = self._resolve_finetune_resume_checkpoint(
                round_index,
                fold_id,
                alignment_cfg,
                alignment_teacher_checkpoint_dir,
            )
            fold_logger = logger.child(f"fold={fold_id}")
            fold_logger.log(
                f"fold start | train_cases={len(train_cases)} holdout_cases={len(holdout_cases)} "
                f"resume={'yes' if previous_ckpt.exists() else 'no'} resume_checkpoint={previous_ckpt if previous_ckpt.exists() else 'n/a'}"
            )
            result = self.backend.train_fold(
                round_index=round_index,
                fold_id=fold_id,
                train_cases=train_cases,
                checkpoint_dir=checkpoint_dir,
                resume_checkpoint=previous_ckpt if previous_ckpt.exists() else None,
                round0=False,
                logger=fold_logger,
                training_csv_path=report_dir / f"fold_{fold_id}_train.csv",
                status_json_path=report_dir / f"fold_{fold_id}_train_status.json",
            )
            fold_decision: dict[str, Any] = {
                "fold_id": fold_id,
                "mode": "candidate",
                "blend_lambda": 1.0,
                "accepted": True,
                "teacher_checkpoint_path": str(previous_ckpt) if previous_ckpt.exists() else None,
                "candidate_checkpoint_path": str(result.checkpoint_path),
            }
            no_regret_enabled = (
                bool(alignment_cfg.get("enabled", False))
                and str(alignment_cfg.get("mode", "")) == "teacher_residual_no_regret"
                and bool(alignment_cfg.get("no_regret_enabled", True))
                and alignment_teacher_oof_dir is not None
            )
            val_cases = [case for case in train_cases if case["case_id"] in set(result.val_case_ids)]
            if no_regret_enabled:
                teacher_val_predictions = self._load_teacher_prediction_dict(
                    [case["case_id"] for case in val_cases],
                    alignment_teacher_oof_dir,
                    alignment_cfg,
                )
                candidate_val_predictions = self.backend.predict_oof_fold(
                    result.checkpoint_path,
                    [{"case_id": case["case_id"], "image": case["image"]} for case in val_cases],
                    logger=fold_logger.child("candidate-validation"),
                    status_json_path=report_dir / f"fold_{fold_id}_candidate_validation_status.json",
                )
                fold_decision = self._select_no_regret_blend(
                    fold_id=fold_id,
                    val_cases=val_cases,
                    teacher_predictions=teacher_val_predictions,
                    candidate_predictions=candidate_val_predictions,
                    alignment_cfg=alignment_cfg,
                )
                fold_decision.update(
                    {
                        "teacher_checkpoint_path": str(previous_ckpt) if previous_ckpt.exists() else None,
                        "candidate_checkpoint_path": str(result.checkpoint_path),
                    }
                )
                if float(fold_decision["blend_lambda"]) == 0.0 and previous_ckpt.exists():
                    copy2(previous_ckpt, result.checkpoint_path)
                    fold_decision["mode"] = "teacher"
                elif float(fold_decision["blend_lambda"]) < 1.0:
                    fold_decision["mode"] = "teacher_candidate_blend"
                else:
                    fold_decision["mode"] = "candidate"
                fold_logger.log(
                    "no-regret selection | "
                    f"lambda={float(fold_decision['blend_lambda']):.3f} mode={fold_decision['mode']} "
                    f"teacher_{self.backend.validation_metric}={fold_decision['teacher_metrics'].get(self.backend.validation_metric, 0.0):.6f} "
                    f"selected_{self.backend.validation_metric}={fold_decision['selected_metrics'].get(self.backend.validation_metric, 0.0):.6f} "
                    f"volume_drift={fold_decision.get('volume_drift_fraction', 0.0):.6f}"
                )
                write_json_atomic(report_dir / f"fold_{fold_id}_selection.json", fold_decision)
            fold_selection_records.append(fold_decision)
            self.store.add_artifact(
                "checkpoint",
                str(result.checkpoint_path),
                sha256_file(result.checkpoint_path),
                round_index=round_index,
                metadata={
                    "fold_id": fold_id,
                    "loss_history": result.loss_history,
                    "best_epoch": result.best_epoch,
                    "best_metric_name": result.best_metric_name,
                    "best_metric_value": result.best_metric_value,
                    "train_case_ids": result.train_case_ids,
                    "val_case_ids": result.val_case_ids,
                    "no_regret_selection": fold_decision,
                },
            )
            self.store.add_artifact(
                "checkpoint_last",
                str(result.last_checkpoint_path),
                sha256_file(result.last_checkpoint_path),
                round_index=round_index,
                metadata={"fold_id": fold_id, "loss_history": result.loss_history},
            )
            self._write_fold_training_log(report_dir, fold_id, result.loss_history)
            if no_regret_enabled:
                teacher_holdout_predictions = self._load_teacher_prediction_dict(
                    [case["case_id"] for case in holdout_cases],
                    alignment_teacher_oof_dir,
                    alignment_cfg,
                )
                blend_lambda = float(fold_decision["blend_lambda"])
                if blend_lambda == 0.0:
                    predictions = teacher_holdout_predictions
                    write_json_atomic(
                        report_dir / f"fold_{fold_id}_inference_status.json",
                        {
                            "stage": "inference",
                            "status": "completed",
                            "source": "teacher_oof",
                            "checkpoint_path": str(previous_ckpt) if previous_ckpt.exists() else None,
                            "num_cases_total": len(holdout_cases),
                            "num_cases_completed": len(holdout_cases),
                        },
                    )
                else:
                    candidate_holdout_predictions = self.backend.predict_oof_fold(
                        result.checkpoint_path,
                        holdout_cases,
                        logger=fold_logger,
                        status_json_path=report_dir / f"fold_{fold_id}_inference_status.json",
                    )
                    predictions = self._blend_prediction_dicts(
                        teacher_holdout_predictions,
                        candidate_holdout_predictions,
                        blend_lambda,
                    )
            else:
                predictions = self.backend.predict_oof_fold(
                    result.checkpoint_path,
                    holdout_cases,
                    logger=fold_logger,
                    status_json_path=report_dir / f"fold_{fold_id}_inference_status.json",
                )
            self._write_fold_inference_log(report_dir, fold_id, list(predictions))
            for case_id, pred in predictions.items():
                oof_path = oof_dir / f"{case_id}.npz"
                np.savez_compressed(oof_path, s=pred["s"], q=pred["q"])
                self.store.add_artifact("oof_prediction", str(oof_path), sha256_file(oof_path), round_index=round_index, case_id=case_id)
            fold_logger.log(f"fold complete | checkpoint={result.checkpoint_path} oof_cases={len(predictions)}")
        write_json_atomic(report_dir / "fold_selection.json", {"folds": fold_selection_records})

        current_uncertainty: dict[str, np.ndarray] = {}
        for case in previous_cases:
            case_id = case["case_id"]
            updated = self.store.get_case(case_id)
            oof_payload = np.load(oof_dir / f"{case_id}.npz")
            soft_target = build_soft_target(load_nifti(Path(updated["current_binary_label_path"])).data.astype(np.uint8), oof_payload["s"], self.config.protocol.alpha)
            uncertainty = compute_uncertainty_from_target(soft_target, self.config.protocol.alpha)
            soft_path = soft_dir / f"{case_id}.npz"
            unc_path = unc_dir / f"{case_id}.npz"
            np.savez_compressed(soft_path, target=soft_target)
            np.savez_compressed(unc_path, uncertainty=uncertainty)
            self._save_postprocessed_mask(round_index=round_index, case_id=case_id, probability=oof_payload["s"], reference_path=Path(updated["current_binary_label_path"]), output_dir=mask_dir)
            updated["current_oof_path"] = str(oof_dir / f"{case_id}.npz")
            updated["current_soft_target_path"] = str(soft_path)
            updated["current_uncertainty_path"] = str(unc_path)
            self.store.upsert_case(updated)
            current_targets[case_id] = soft_target
            current_uncertainty[case_id] = uncertainty
            self.store.add_artifact("soft_target", str(soft_path), sha256_file(soft_path), round_index=round_index, case_id=case_id)
            self.store.add_artifact("uncertainty", str(unc_path), sha256_file(unc_path), round_index=round_index, case_id=case_id)

        latest_cases = [self._load_case_payload(case) for case in self.store.list_cases()]
        metrics = compute_round_summary(
            latest_cases,
            round_record["routine_ids"],
            round_record["audit_ids"],
            previous_targets,
            current_targets,
            previous_predictions,
            current_uncertainty,
            review_records,
            previous_binary_labels=previous_binary_labels,
            high_uncertainty_threshold=float(self.config.protocol.uncertainty.get("high_threshold", 0.5)),
        )
        oof_metrics = self._compute_oof_metrics(latest_cases)
        self._write_oof_reports(report_dir, oof_metrics)
        self._rewrite_fold_inference_logs(report_dir, fold_case_ids, oof_metrics["fold_rows"])
        metrics["oof"] = oof_metrics["summary"]
        round_metrics_history.append(metrics)
        breadth_threshold = min(1.0, max((3.0 * round_record["budget"]) / max(len(latest_cases), 1), 0.2))
        stop_state = compute_stop_state(round_metrics_history, metrics["cov"], breadth_threshold, self.config.protocol.stop)
        self.store.upsert_round_metrics(round_index, metrics, metadata={"oof": oof_metrics["summary"]})
        self.store.upsert_round(round_index, {**round_record, "status": "completed", "metrics": metrics, "stop_state": stop_state})
        logger.log(
            f"finalize complete | round={round_index} macro_dice_raw={oof_metrics['summary']['macro_dice_raw']:.6f} "
            f"macro_dice_post={oof_metrics['summary']['macro_dice_postprocessed']:.6f} should_stop={stop_state['should_stop']}"
        )

    def diagnose_revision_policy(self, round_index: int) -> None:
        logger = self._create_run_logger("diagnose-revision-policy", round_index=round_index)
        round_record = self.store.get_round(round_index)
        if round_record is None:
            raise RuntimeError(f"Missing round {round_index}")
        revision_cfg = dict(self.config.model.training.get("revision_policy", {}))
        if str(revision_cfg.get("mode", "sgra")) != "sgra":
            raise RuntimeError(f"Unsupported revision_policy.mode: {revision_cfg.get('mode')}")
        candidate_cfg = dict(revision_cfg.get("candidate", {}))
        oracle_cfg = dict(revision_cfg.get("oracle", {}))
        base_oof_dir, base_oof_source = self._resolve_revision_base_round1_oof_dir(round_index, revision_cfg)
        base_unc_dir, base_unc_source = self._resolve_revision_base_round1_uncertainty_dir(round_index, revision_cfg)
        report_dir = ensure_dir(self.config.workspace_root / "reports" / f"round_{round_index}")
        review_stats_map = {row["case_id"]: row for row in self.store.list_review_stats(round_index)}
        previous_cases = [self._load_case_payload(case) for case in self.store.list_cases()]
        revision_cases: list[RevisionCase] = []
        for case in previous_cases:
            case_id = case["case_id"]
            p0 = case["previous_oof"].astype(np.float32)
            q0 = self._load_uncertainty_map(case).astype(np.float32)
            p_base, q_from_base_oof = self._load_revision_oof_payload(base_oof_dir, case_id, revision_cfg)
            q_base = self._load_revision_uncertainty(base_unc_dir, case_id, q_from_base_oof)
            if not bool(revision_cfg.get("use_round0", True)):
                p0 = p_base.copy()
                q0 = np.zeros_like(q_base, dtype=np.float32)
            previous_binary = case["current_binary_label"].astype(np.uint8)
            current_binary = previous_binary.copy()
            role = None
            if case_id in round_record["routine_ids"]:
                role = "routine"
                stats_row = review_stats_map.get(case_id)
                if stats_row is None or not stats_row.get("routine_final_label_path"):
                    raise RuntimeError(f"Missing routine final label for {case_id}")
                current_binary = import_review_label(Path(stats_row["routine_final_label_path"]))["binary"]
            elif case_id in round_record["audit_ids"]:
                role = "audit"
                stats_row = review_stats_map.get(case_id)
                if stats_row is None or not stats_row.get("audit_final_label_path"):
                    raise RuntimeError(f"Missing audit final label for {case_id}")
                current_binary = import_review_label(Path(stats_row["audit_final_label_path"]))["binary"]
            revision_cases.append(
                make_revision_case(
                    case_id=case_id,
                    fold_id=int(case["fold_id"]),
                    role=role,
                    image=self.backend.preprocess_image(case["image"].astype(np.float32)),
                    p_round0=p0,
                    q_round0=q0,
                    p_base=p_base,
                    q_base=q_base,
                    y_old=previous_binary,
                    y_final=current_binary,
                    candidate_cfg=candidate_cfg,
                )
            )
        oracle_rows = [oracle_case_summary(case, candidate_cfg) for case in revision_cases if case.role is not None] if bool(oracle_cfg.get("enabled", True)) else []
        self._write_revision_oracle_report(report_dir, oracle_rows, oracle_cfg)
        self._write_revision_baseline_comparison(report_dir, revision_cases)
        logger.log(
            f"diagnose revision policy complete | round={round_index} base_round1_oof={base_oof_source} "
            f"base_round1_uncertainty={base_unc_source} reviewed={len(oracle_rows)}"
        )

    def _finalize_round_revision_policy(
        self,
        round_index: int,
        logger: RunLogger,
        round_record: dict[str, Any],
        revision_cfg: dict[str, Any],
    ) -> None:
        previous_cases = [self._load_case_payload(case) for case in self.store.list_cases()]
        logger.log(
            f"revision policy finalize start | round={round_index} cases={len(previous_cases)} "
            f"mode={revision_cfg.get('mode', 'sgra')}"
        )
        if bool(self.config.model.training.get("alignment", {}).get("enabled", False)):
            logger.log("revision policy enabled | ignoring training.alignment for this finalize run")
        if str(revision_cfg.get("mode", "sgra")) != "sgra":
            raise RuntimeError(f"Unsupported revision_policy.mode: {revision_cfg.get('mode')}")
        candidate_cfg = dict(revision_cfg.get("candidate", {}))
        oracle_cfg = dict(revision_cfg.get("oracle", {}))
        selector_cfg = dict(revision_cfg.get("component_selector", {}))
        adapter_cfg = dict(revision_cfg.get("adapter", {}))
        accept_cfg = dict(revision_cfg.get("accept", {}))
        outputs_cfg = dict(revision_cfg.get("outputs", {}))
        base_oof_dir, base_oof_source = self._resolve_revision_base_round1_oof_dir(round_index, revision_cfg)
        base_unc_dir, base_unc_source = self._resolve_revision_base_round1_uncertainty_dir(round_index, revision_cfg)
        logger.log(f"revision policy base | base_round1_oof={base_oof_source} base_round1_uncertainty={base_unc_source}")

        review_stats_map = {row["case_id"]: row for row in self.store.list_review_stats(round_index)}
        raw_dir = ensure_dir(self.config.workspace_root / "artifacts" / "labels" / "raw" / f"round_{round_index}")
        bin_dir = ensure_dir(self.config.workspace_root / "artifacts" / "labels" / "binary" / f"round_{round_index}")
        oof_dir = ensure_dir(self.config.workspace_root / "artifacts" / "oof" / f"round_{round_index}")
        mask_dir = ensure_dir(self.config.workspace_root / "artifacts" / "masks" / f"round_{round_index}")
        soft_dir = ensure_dir(self.config.workspace_root / "artifacts" / "soft_targets" / f"round_{round_index}")
        unc_dir = ensure_dir(self.config.workspace_root / "artifacts" / "uncertainty" / f"round_{round_index}")
        adapter_dir = ensure_dir(self.config.workspace_root / "artifacts" / "adapters" / f"round_{round_index}")
        report_dir = ensure_dir(self.config.workspace_root / "reports" / f"round_{round_index}")
        if bool(revision_cfg.get("strict_oof", True)) and base_oof_dir.resolve() == oof_dir.resolve():
            raise RuntimeError("Revision policy strict_oof forbids using the round output directory as base_round1")

        previous_targets: dict[str, np.ndarray] = {}
        current_targets: dict[str, np.ndarray] = {}
        previous_predictions: dict[str, np.ndarray] = {}
        previous_binary_labels: dict[str, np.ndarray] = {}
        current_uncertainty: dict[str, np.ndarray] = {}
        review_records: dict[str, dict[str, Any]] = {}
        revision_cases: list[RevisionCase] = []

        for case in previous_cases:
            case_id = case["case_id"]
            previous_targets[case_id] = np.load(case["current_soft_target_path"])["target"].astype(np.float32)
            p0 = case["previous_oof"].astype(np.float32)
            q0 = self._load_uncertainty_map(case).astype(np.float32)
            p_base, q_from_base_oof = self._load_revision_oof_payload(base_oof_dir, case_id, revision_cfg)
            q_base = self._load_revision_uncertainty(base_unc_dir, case_id, q_from_base_oof)
            if not bool(revision_cfg.get("use_round0", True)):
                p0 = p_base.copy()
                q0 = np.zeros_like(q_base, dtype=np.float32)
            previous_predictions[case_id] = p0
            previous_binary = case["current_binary_label"].astype(np.uint8)
            previous_binary_labels[case_id] = previous_binary
            current_raw = load_nifti(Path(case["current_raw_label_path"]))
            current_binary = previous_binary.copy()
            role = None
            if case_id in round_record["routine_ids"]:
                role = "routine"
                stats_row = review_stats_map.get(case_id)
                if stats_row is None or not stats_row.get("routine_final_label_path"):
                    raise RuntimeError(f"Missing routine review stats for {case_id}")
                imported = import_review_label(Path(stats_row["routine_final_label_path"]))
                current_raw = load_nifti(Path(stats_row["routine_final_label_path"]))
                current_binary = imported["binary"]
                review_records[case_id] = {**stats_row, "previous_binary": previous_binary, "final_binary": current_binary}
            elif case_id in round_record["audit_ids"]:
                role = "audit"
                stats_row = review_stats_map.get(case_id)
                if stats_row is None or not stats_row.get("audit_anchor_label_path") or not stats_row.get("audit_final_label_path"):
                    raise RuntimeError(f"Missing audit review stats for {case_id}")
                anchor = import_review_label(Path(stats_row["audit_anchor_label_path"]))
                final = import_review_label(Path(stats_row["audit_final_label_path"]))
                current_raw = load_nifti(Path(stats_row["audit_final_label_path"]))
                current_binary = final["binary"]
                review_records[case_id] = {
                    **stats_row,
                    "previous_binary": previous_binary,
                    "anchor_binary": anchor["binary"],
                    "final_binary": current_binary,
                }
            raw_path = raw_dir / f"{case_id}.nii.gz"
            bin_path = bin_dir / f"{case_id}.nii.gz"
            save_nifti(raw_path, current_raw.data.astype(np.int16), current_raw.affine, current_raw.header, np.int16)
            save_nifti(bin_path, current_binary, current_raw.affine, current_raw.header, np.uint8)
            updated = self.store.get_case(case_id)
            if role is not None:
                updated["review_count"] = int(updated["review_count"]) + 1
                updated["last_review_round"] = round_index
                anchor_dice = float(review_records[case_id].get("anchor_assisted_dice") or 1.0)
                updated["earliest_eligible_round"] = compute_review_reentry(anchor_dice, self.config.protocol.review, round_index, role)
            updated["current_raw_label_path"] = str(raw_path)
            updated["current_binary_label_path"] = str(bin_path)
            self.store.upsert_case(updated)
            revision_cases.append(
                make_revision_case(
                    case_id=case_id,
                    fold_id=int(case["fold_id"]),
                    role=role,
                    image=self.backend.preprocess_image(case["image"].astype(np.float32)),
                    p_round0=p0,
                    q_round0=q0,
                    p_base=p_base,
                    q_base=q_base,
                    y_old=previous_binary,
                    y_final=current_binary,
                    candidate_cfg=candidate_cfg,
                )
            )

        oracle_rows = [oracle_case_summary(case, candidate_cfg) for case in revision_cases if case.role is not None] if bool(oracle_cfg.get("enabled", True)) else []
        self._write_revision_oracle_report(report_dir, oracle_rows, oracle_cfg)

        device = self.backend.resolve_device()
        fold_case_ids: dict[int, list[str]] = {}
        fold_records: list[dict[str, Any]] = []
        component_rows: list[dict[str, Any]] = []
        predictions: dict[str, dict[str, np.ndarray]] = {}
        for fold_id in range(1, self.config.protocol.folds + 1):
            fold_logger = logger.child(f"fold={fold_id}")
            train_cases = [case for case in revision_cases if case.fold_id != fold_id]
            holdout_cases = [case for case in revision_cases if case.fold_id == fold_id]
            fold_case_ids[fold_id] = [case.case_id for case in holdout_cases]
            fold_logger.log(f"revision fold start | train_cases={len(train_cases)} holdout_cases={len(holdout_cases)} device={device.type}")
            selector_model = train_component_selector(train_cases, candidate_cfg, selector_cfg) if bool(selector_cfg.get("train", True)) else {"constant_probability": 0.0}
            selector_path = adapter_dir / f"fold_{fold_id}_selector.json"
            write_selector_model(selector_path, selector_model)
            self.store.add_artifact("revision_policy", str(selector_path), sha256_file(selector_path), round_index=round_index, metadata={"fold_id": fold_id, "kind": "component_selector"})
            adapter_path = adapter_dir / f"fold_{fold_id}.pt"
            adapter_status: dict[str, Any] | None = None
            if bool(adapter_cfg.get("enabled", True)) and bool(adapter_cfg.get("train", True)):
                adapter_status = train_adapter(
                    train_cases,
                    adapter_cfg,
                    adapter_path,
                    device=device,
                    seed=self.config.protocol.seed + round_index * 100 + fold_id,
                )
                self.store.add_artifact("revision_policy", str(adapter_path), sha256_file(adapter_path), round_index=round_index, metadata={"fold_id": fold_id, "kind": "sgra_adapter"})
            else:
                adapter_path.write_text(json.dumps({"disabled": True, "fold_id": fold_id}), encoding="utf-8")
                self.store.add_artifact("revision_policy", str(adapter_path), sha256_file(adapter_path), round_index=round_index, metadata={"fold_id": fold_id, "kind": "adapter_disabled"})
            fold_component_count = 0
            fold_applied_count = 0
            for case in holdout_cases:
                probability = case.p_base.copy()
                if adapter_status is not None:
                    probability = predict_adapter(case, adapter_path, adapter_cfg, device=device)
                if bool(selector_cfg.get("enabled", True)):
                    selector_apply_cfg = dict(selector_cfg)
                    if not bool(accept_cfg.get("component_no_regret", True)):
                        selector_apply_cfg["min_predicted_gain"] = 0.0
                    probability, records = apply_component_selector(case, selector_model, candidate_cfg, selector_apply_cfg, probability)
                    for row in records:
                        row["fold_id"] = fold_id
                        component_rows.append(row)
                    fold_component_count += len(records)
                    fold_applied_count += sum(1 for row in records if row["applied"])
                probability, guard = apply_case_guard(
                    probability,
                    case.p_base,
                    accept_cfg,
                    threshold=float(self.config.model.postprocessing.get("threshold", self.config.model.inference.get("threshold", 0.5))),
                )
                component_rows.append({"case_id": case.case_id, "fold_id": fold_id, "action": "case_guard", **guard})
                q = np.maximum(case.q_base, np.abs(probability - case.p_base)).astype(np.float32)
                predictions[case.case_id] = {"s": probability.astype(np.float32), "q": q}
                oof_path = oof_dir / f"{case.case_id}.npz"
                np.savez_compressed(oof_path, s=predictions[case.case_id]["s"], q=predictions[case.case_id]["q"])
                self.store.add_artifact("oof_prediction", str(oof_path), sha256_file(oof_path), round_index=round_index, case_id=case.case_id)
            fold_record = {
                "fold_id": fold_id,
                "mode": "revision_policy_sgra",
                "train_case_ids": [case.case_id for case in train_cases],
                "holdout_case_ids": [case.case_id for case in holdout_cases],
                "selector": {key: selector_model.get(key) for key in ("num_candidates", "num_positive", "expected_positive_gain", "constant_probability") if key in selector_model},
                "adapter": adapter_status or {"disabled": True},
                "component_candidates": fold_component_count,
                "components_applied": fold_applied_count,
            }
            fold_records.append(fold_record)
            write_json_atomic(report_dir / f"fold_{fold_id}_revision_policy_status.json", fold_record)
            fold_logger.log(f"revision fold complete | components={fold_component_count} applied={fold_applied_count} oof_cases={len(holdout_cases)}")
        write_json_atomic(report_dir / "revision_policy_fold_summary.json", {"folds": fold_records})
        write_json_atomic(report_dir / "fold_selection.json", {"folds": fold_records})
        self._write_component_correction_metrics(report_dir, component_rows)

        for case in previous_cases:
            case_id = case["case_id"]
            updated = self.store.get_case(case_id)
            pred = predictions[case_id]
            soft_target = build_soft_target(load_nifti(Path(updated["current_binary_label_path"])).data.astype(np.uint8), pred["s"], self.config.protocol.alpha)
            uncertainty = compute_uncertainty_from_target(soft_target, self.config.protocol.alpha)
            soft_path = soft_dir / f"{case_id}.npz"
            unc_path = unc_dir / f"{case_id}.npz"
            np.savez_compressed(soft_path, target=soft_target)
            np.savez_compressed(unc_path, uncertainty=uncertainty)
            self._save_postprocessed_mask(round_index=round_index, case_id=case_id, probability=pred["s"], reference_path=Path(updated["current_binary_label_path"]), output_dir=mask_dir)
            updated["current_oof_path"] = str(oof_dir / f"{case_id}.npz")
            updated["current_soft_target_path"] = str(soft_path)
            updated["current_uncertainty_path"] = str(unc_path)
            self.store.upsert_case(updated)
            current_targets[case_id] = soft_target
            current_uncertainty[case_id] = uncertainty
            self.store.add_artifact("soft_target", str(soft_path), sha256_file(soft_path), round_index=round_index, case_id=case_id)
            self.store.add_artifact("uncertainty", str(unc_path), sha256_file(unc_path), round_index=round_index, case_id=case_id)

        latest_cases = [self._load_case_payload(case) for case in self.store.list_cases()]
        metrics = compute_round_summary(
            latest_cases,
            round_record["routine_ids"],
            round_record["audit_ids"],
            previous_targets,
            current_targets,
            previous_predictions,
            current_uncertainty,
            review_records,
            previous_binary_labels=previous_binary_labels,
            high_uncertainty_threshold=float(self.config.protocol.uncertainty.get("high_threshold", 0.5)),
        )
        oof_metrics = self._compute_oof_metrics(latest_cases)
        self._write_oof_reports(report_dir, oof_metrics)
        self._rewrite_fold_inference_logs(report_dir, fold_case_ids, oof_metrics["fold_rows"])
        if bool(outputs_cfg.get("write_baseline_comparison", True)):
            self._write_revision_baseline_comparison(report_dir, revision_cases)
        if bool(outputs_cfg.get("write_hitl_corrected_system", True)):
            self._write_hitl_corrected_summary(report_dir, latest_cases, round_record)
        metrics["oof"] = oof_metrics["summary"]
        metrics["revision_policy"] = {
            "mode": revision_cfg.get("mode", "sgra"),
            "base_round1_oof": base_oof_source,
            "base_round1_uncertainty": base_unc_source,
        }
        round_metrics_history = []
        for existing_round in range(1, round_index):
            existing = self.store.get_round(existing_round)
            if existing is not None and existing["metrics"]:
                round_metrics_history.append(existing["metrics"])
        round_metrics_history.append(metrics)
        breadth_threshold = min(1.0, max((3.0 * round_record["budget"]) / max(len(latest_cases), 1), 0.2))
        stop_state = compute_stop_state(round_metrics_history, metrics["cov"], breadth_threshold, self.config.protocol.stop)
        self.store.upsert_round_metrics(round_index, metrics, metadata={"oof": oof_metrics["summary"], "revision_policy": metrics["revision_policy"]})
        self.store.upsert_round(round_index, {**round_record, "status": "completed", "metrics": metrics, "stop_state": stop_state})
        logger.log(
            f"revision policy finalize complete | round={round_index} macro_dice_raw={oof_metrics['summary']['macro_dice_raw']:.6f} "
            f"macro_dice_post={oof_metrics['summary']['macro_dice_postprocessed']:.6f} should_stop={stop_state['should_stop']}"
        )

    def _resolve_revision_base_round1_oof_dir(self, round_index: int, revision_cfg: dict[str, Any]) -> tuple[Path, str]:
        configured = revision_cfg.get("base_round1_oof_dir")
        if configured:
            path = self._resolve_config_path(configured)
            if path.exists():
                return path, str(path)
            if bool(revision_cfg.get("require_base_round1", True)):
                raise RuntimeError(f"Configured revision base_round1_oof_dir does not exist: {path}")
        base_value = revision_cfg.get("base_round1")
        if base_value and str(base_value) not in {"auto", "baseline_round1"}:
            path = self._resolve_config_path(base_value)
            if path.exists():
                return path, str(path)
            if bool(revision_cfg.get("require_base_round1", True)):
                raise RuntimeError(f"Configured revision base_round1 path does not exist: {path}")
        for archive in self._round_archive_candidates(round_index):
            candidate = archive / f"artifacts__oof__round_{round_index}"
            if candidate.exists():
                return candidate, str(candidate)
        fallback = self.config.workspace_root / "artifacts" / "oof" / f"round_{round_index - 1}"
        if fallback.exists() and not bool(revision_cfg.get("require_base_round1", True)):
            return fallback, str(fallback)
        raise RuntimeError(f"Missing archived base round {round_index} OOF for revision policy")

    def _resolve_revision_base_round1_uncertainty_dir(self, round_index: int, revision_cfg: dict[str, Any]) -> tuple[Path | None, str]:
        configured = revision_cfg.get("base_round1_uncertainty_dir")
        if configured:
            path = self._resolve_config_path(configured)
            if path.exists():
                return path, str(path)
        for archive in self._round_archive_candidates(round_index):
            candidate = archive / f"artifacts__uncertainty__round_{round_index}"
            if candidate.exists():
                return candidate, str(candidate)
        return None, "base_oof_q"

    def _resolve_revision_base_round1_checkpoint_dir(self, round_index: int, revision_cfg: dict[str, Any], require: bool | None = None) -> tuple[Path | None, str]:
        require_base = bool(revision_cfg.get("require_base_round1", True)) if require is None else bool(require)
        configured = revision_cfg.get("base_round1_checkpoint_dir")
        if configured:
            path = self._resolve_config_path(configured)
            if path.exists():
                return path, str(path)
            if require_base:
                raise RuntimeError(f"Configured revision base_round1_checkpoint_dir does not exist: {path}")
        for archive in self._round_archive_candidates(round_index):
            candidate = archive / f"artifacts__checkpoints__round_{round_index}"
            if candidate.exists():
                return candidate, str(candidate)
        fallback = self.config.workspace_root / "artifacts" / "checkpoints" / f"round_{round_index - 1}"
        if fallback.exists() and not require_base:
            return fallback, str(fallback)
        if require_base:
            raise RuntimeError(f"Missing archived base round {round_index} checkpoints for revision policy")
        return None, "missing"

    def _load_revision_oof_payload(self, base_oof_dir: Path, case_id: str, revision_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        path = base_oof_dir / f"{case_id}.npz"
        if not path.exists():
            if bool(revision_cfg.get("require_base_round1", True)):
                raise RuntimeError(f"Missing base round1 OOF for {case_id}: {path}")
            raise FileNotFoundError(path)
        payload = np.load(path)
        probability = payload["s"].astype(np.float32)
        q = payload["q"].astype(np.float32) if "q" in payload else np.zeros_like(probability, dtype=np.float32)
        return probability, q

    def _load_revision_uncertainty(self, base_unc_dir: Path | None, case_id: str, fallback_q: np.ndarray) -> np.ndarray:
        if base_unc_dir is not None:
            path = base_unc_dir / f"{case_id}.npz"
            if path.exists():
                payload = np.load(path)
                if "uncertainty" in payload:
                    return payload["uncertainty"].astype(np.float32)
                if "q" in payload:
                    return payload["q"].astype(np.float32)
        return fallback_q.astype(np.float32)

    def _write_revision_oracle_report(self, report_dir: Path, rows: list[dict[str, Any]], oracle_cfg: dict[str, Any] | None = None) -> None:
        oracle_cfg = oracle_cfg or {}
        path = report_dir / "revision_oracle_summary.csv"
        fieldnames = [
            "case_id",
            "fold_id",
            "role",
            "base_dice",
            "candidate_only_oracle_dice",
            "candidate_only_oracle_gain",
            "component_oracle_dice",
            "component_oracle_gain",
            "positive_oracle_components",
            "base_added_voxels",
            "base_removed_voxels",
            "review_added_voxels",
            "review_removed_voxels",
            "base_add_coverage",
            "base_remove_coverage",
            "review_add_coverage",
            "review_remove_coverage",
            "action_region_voxels",
            "action_region_fraction",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if rows:
            summary = {
                "num_reviewed_cases": len(rows),
                "mean_base_add_coverage": float(np.mean([float(row["base_add_coverage"]) for row in rows])),
                "mean_base_remove_coverage": float(np.mean([float(row["base_remove_coverage"]) for row in rows])),
                "mean_candidate_only_oracle_gain": float(np.mean([float(row["candidate_only_oracle_gain"]) for row in rows])),
                "mean_component_oracle_gain": float(np.mean([float(row["component_oracle_gain"]) for row in rows])),
                "min_add_coverage": float(oracle_cfg.get("min_add_coverage", 0.70)),
                "min_remove_coverage": float(oracle_cfg.get("min_remove_coverage", 0.70)),
                "coverage_pass": bool(
                    float(np.mean([float(row["base_add_coverage"]) for row in rows])) >= float(oracle_cfg.get("min_add_coverage", 0.70))
                    and float(np.mean([float(row["base_remove_coverage"]) for row in rows])) >= float(oracle_cfg.get("min_remove_coverage", 0.70))
                ),
            }
        else:
            summary = {
                "num_reviewed_cases": 0,
                "mean_base_add_coverage": 0.0,
                "mean_base_remove_coverage": 0.0,
                "mean_candidate_only_oracle_gain": 0.0,
                "mean_component_oracle_gain": 0.0,
                "min_add_coverage": float(oracle_cfg.get("min_add_coverage", 0.70)),
                "min_remove_coverage": float(oracle_cfg.get("min_remove_coverage", 0.70)),
                "coverage_pass": False,
            }
        write_json_atomic(report_dir / "revision_oracle_summary.json", summary)

    def _write_component_correction_metrics(self, report_dir: Path, rows: list[dict[str, Any]]) -> None:
        path = report_dir / "component_correction_metrics.csv"
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["case_id", "fold_id", "action"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _probability_metrics_for_cases(self, cases: list[dict[str, Any]], probabilities: dict[str, np.ndarray]) -> dict[str, Any]:
        threshold = float(self.config.model.postprocessing.get("threshold", self.config.model.inference.get("threshold", 0.5)))
        rows: list[dict[str, Any]] = []
        for case in cases:
            case_id = case["case_id"]
            probability = probabilities[case_id].astype(np.float32)
            target = case["current_binary_label"].astype(np.uint8)
            pred_raw = (probability >= threshold).astype(np.uint8)
            pred_post = self._postprocess_probability(probability)
            rows.append(
                {
                    "case_id": case_id,
                    "dice_raw": float(dice_score(pred_raw, target)),
                    "dice_postprocessed": float(dice_score(pred_post, target)),
                    "intersection_raw": int(np.logical_and(pred_raw.astype(bool), target.astype(bool)).sum()),
                    "intersection_postprocessed": int(np.logical_and(pred_post.astype(bool), target.astype(bool)).sum()),
                    "gt_positive_voxels": int(target.sum()),
                    "pred_positive_voxels_raw": int(pred_raw.sum()),
                    "pred_positive_voxels_postprocessed": int(pred_post.sum()),
                }
            )
        return {"summary": self._aggregate_prediction_rows(rows), "case_rows": rows}

    def _write_revision_baseline_comparison(self, report_dir: Path, revision_cases: list[RevisionCase]) -> None:
        cases = [
            {
                "case_id": case.case_id,
                "current_binary_label": case.y_final,
            }
            for case in revision_cases
        ]
        metrics = self._probability_metrics_for_cases(cases, {case.case_id: case.p_base for case in revision_cases})
        write_json_atomic(report_dir / "baseline_round1_summary.json", metrics["summary"])
        with (report_dir / "baseline_round1_case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "case_id",
                "dice_raw",
                "dice_postprocessed",
                "intersection_raw",
                "intersection_postprocessed",
                "gt_positive_voxels",
                "pred_positive_voxels_raw",
                "pred_positive_voxels_postprocessed",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics["case_rows"])

    def _write_hitl_corrected_summary(self, report_dir: Path, latest_cases: list[dict[str, Any]], round_record: dict[str, Any]) -> None:
        selected = set(round_record.get("routine_ids", [])) | set(round_record.get("audit_ids", []))
        probabilities: dict[str, np.ndarray] = {}
        for case in latest_cases:
            case_id = case["case_id"]
            if case_id in selected:
                probabilities[case_id] = case["current_binary_label"].astype(np.float32)
            else:
                probabilities[case_id] = case["previous_oof"].astype(np.float32)
        metrics = self._probability_metrics_for_cases(latest_cases, probabilities)
        metrics["summary"]["reviewed_cases_use_final_label"] = len(selected)
        metrics["summary"]["unreviewed_cases_use_model_prediction"] = max(len(latest_cases) - len(selected), 0)
        write_json_atomic(report_dir / "hitl_corrected_summary.json", metrics["summary"])
        with (report_dir / "hitl_corrected_case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "case_id",
                "dice_raw",
                "dice_postprocessed",
                "intersection_raw",
                "intersection_postprocessed",
                "gt_positive_voxels",
                "pred_positive_voxels_raw",
                "pred_positive_voxels_postprocessed",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics["case_rows"])

    def _resolve_alignment_teacher_oof_dir(self, round_index: int, alignment_cfg: dict[str, Any]) -> tuple[Path | None, str]:
        if not bool(alignment_cfg.get("enabled", False)):
            return None, "disabled"
        configured = alignment_cfg.get("teacher_oof_dir")
        if configured:
            path = self._resolve_config_path(configured)
            if path.exists():
                return path, str(path)
            if bool(alignment_cfg.get("require_teacher", False)):
                raise RuntimeError(f"Configured alignment teacher_oof_dir does not exist: {path}")
        teacher = str(alignment_cfg.get("teacher", "auto"))
        if teacher in {"auto", "baseline_round1"}:
            for archive in self._round_archive_candidates(round_index):
                candidate = archive / f"artifacts__oof__round_{round_index}"
                if candidate.exists():
                    return candidate, str(candidate)
            if teacher == "baseline_round1" and bool(alignment_cfg.get("require_teacher", False)):
                raise RuntimeError(f"Missing archived baseline round {round_index} OOF teacher")
        previous = self.config.workspace_root / "artifacts" / "oof" / f"round_{round_index - 1}"
        if previous.exists():
            return previous, str(previous)
        if bool(alignment_cfg.get("require_teacher", False)):
            raise RuntimeError(f"Missing previous-round OOF teacher: {previous}")
        return None, "previous_oof_in_memory"

    def _resolve_alignment_teacher_checkpoint_dir(self, round_index: int, alignment_cfg: dict[str, Any]) -> tuple[Path | None, str]:
        if not bool(alignment_cfg.get("enabled", False)) or not bool(alignment_cfg.get("resume_from_teacher_checkpoint", False)):
            return None, "disabled"
        configured = alignment_cfg.get("teacher_checkpoint_dir")
        if configured:
            path = self._resolve_config_path(configured)
            if path.exists():
                return path, str(path)
            if bool(alignment_cfg.get("require_teacher_checkpoint", False)):
                raise RuntimeError(f"Configured alignment teacher_checkpoint_dir does not exist: {path}")
        for archive in self._round_archive_candidates(round_index):
            candidate = archive / f"artifacts__checkpoints__round_{round_index}"
            if candidate.exists():
                return candidate, str(candidate)
        previous = self.config.workspace_root / "artifacts" / "checkpoints" / f"round_{round_index - 1}"
        if previous.exists():
            return previous, str(previous)
        if bool(alignment_cfg.get("require_teacher_checkpoint", False)):
            raise RuntimeError(f"Missing archived baseline round {round_index} checkpoint teacher")
        return None, "previous_round_checkpoint"

    def _resolve_config_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        workspace_path = self.config.workspace_root / path
        if workspace_path.exists():
            return workspace_path
        return self.config.project_root / path

    def _round_archive_candidates(self, round_index: int) -> list[Path]:
        candidates: list[Path] = []
        for base in {self.config.workspace_root, self.config.workspace_root.parent}:
            if base.exists():
                candidates.extend(path for path in base.glob(f"archive_round{round_index}_previous_*") if path.is_dir())
                candidates.extend(path for path in base.glob(f"archive_round{round_index}_*") if path.is_dir())
        return sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)

    def _load_alignment_teacher_probability(
        self,
        case_id: str,
        teacher_oof_dir: Path | None,
        fallback_probability: np.ndarray,
        alignment_cfg: dict[str, Any],
    ) -> np.ndarray:
        if not bool(alignment_cfg.get("enabled", False)):
            return fallback_probability.astype(np.float32)
        if teacher_oof_dir is not None:
            path = teacher_oof_dir / f"{case_id}.npz"
            if path.exists():
                return np.load(path)["s"].astype(np.float32)
            if bool(alignment_cfg.get("require_teacher", False)):
                raise RuntimeError(f"Missing alignment teacher OOF for {case_id}: {path}")
        return fallback_probability.astype(np.float32)

    def _resolve_finetune_resume_checkpoint(
        self,
        round_index: int,
        fold_id: int,
        alignment_cfg: dict[str, Any],
        teacher_checkpoint_dir: Path | None,
    ) -> Path:
        if bool(alignment_cfg.get("enabled", False)) and bool(alignment_cfg.get("resume_from_teacher_checkpoint", False)) and teacher_checkpoint_dir is not None:
            candidate = teacher_checkpoint_dir / f"fold_{fold_id}.pt"
            if candidate.exists():
                return candidate
            if bool(alignment_cfg.get("require_teacher_checkpoint", False)):
                raise RuntimeError(f"Missing alignment teacher checkpoint for fold {fold_id}: {candidate}")
        return self.config.workspace_root / "artifacts" / "checkpoints" / f"round_{round_index - 1}" / f"fold_{fold_id}.pt"

    def _load_teacher_prediction_dict(
        self,
        case_ids: list[str],
        teacher_oof_dir: Path,
        alignment_cfg: dict[str, Any],
    ) -> dict[str, dict[str, np.ndarray]]:
        predictions: dict[str, dict[str, np.ndarray]] = {}
        for case_id in case_ids:
            path = teacher_oof_dir / f"{case_id}.npz"
            if not path.exists():
                if bool(alignment_cfg.get("require_teacher", False)):
                    raise RuntimeError(f"Missing alignment teacher OOF for {case_id}: {path}")
                continue
            payload = np.load(path)
            variance = payload["q"].astype(np.float32) if "q" in payload else np.zeros_like(payload["s"], dtype=np.float32)
            predictions[case_id] = {"s": payload["s"].astype(np.float32), "q": variance}
        return predictions

    def _blend_prediction_dicts(
        self,
        teacher_predictions: dict[str, dict[str, np.ndarray]],
        candidate_predictions: dict[str, dict[str, np.ndarray]],
        blend_lambda: float,
    ) -> dict[str, dict[str, np.ndarray]]:
        blend = float(np.clip(blend_lambda, 0.0, 1.0))
        output: dict[str, dict[str, np.ndarray]] = {}
        for case_id, teacher in teacher_predictions.items():
            candidate = candidate_predictions[case_id]
            output[case_id] = {
                "s": ((1.0 - blend) * teacher["s"] + blend * candidate["s"]).astype(np.float32),
                "q": ((1.0 - blend) * teacher.get("q", 0.0) + blend * candidate.get("q", 0.0)).astype(np.float32),
            }
        return output

    def _select_no_regret_blend(
        self,
        fold_id: int,
        val_cases: list[dict[str, Any]],
        teacher_predictions: dict[str, dict[str, np.ndarray]],
        candidate_predictions: dict[str, dict[str, np.ndarray]],
        alignment_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        lambdas = [float(value) for value in alignment_cfg.get("candidate_blend_lambdas", [0.0, 1.0])]
        if 0.0 not in lambdas:
            lambdas.insert(0, 0.0)
        lambdas = sorted({float(np.clip(value, 0.0, 1.0)) for value in lambdas})
        teacher_metrics = self._metrics_for_prediction_dict(val_cases, teacher_predictions)
        metric_name = str(alignment_cfg.get("selection_metric", self.backend.validation_metric))
        accept_margin = float(alignment_cfg.get("accept_margin", 0.0))
        max_volume_drift = float(alignment_cfg.get("max_volume_drift_fraction", 1.0))
        tiny_guard = bool(alignment_cfg.get("tiny_lesion_guard", False))
        tiny_margin = float(alignment_cfg.get("tiny_accept_margin", 0.0))
        tiny_fp_margin = float(alignment_cfg.get("tiny_fp_margin", 0.0))
        candidate_records: list[dict[str, Any]] = []
        selected_record: dict[str, Any] | None = None
        teacher_score = float(teacher_metrics.get(metric_name, 0.0))
        for blend in lambdas:
            predictions = teacher_predictions if blend == 0.0 else self._blend_prediction_dicts(teacher_predictions, candidate_predictions, blend)
            metrics = self._metrics_for_prediction_dict(val_cases, predictions)
            volume_drift = self._volume_drift_fraction(metrics, teacher_metrics)
            tiny_ok = True
            if tiny_guard and int(teacher_metrics.get("tiny_num_cases", 0)) > 0:
                tiny_ok = (
                    float(metrics.get("tiny_macro_dice_postprocessed", 0.0))
                    >= float(teacher_metrics.get("tiny_macro_dice_postprocessed", 0.0)) - tiny_margin
                    and float(metrics.get("tiny_fp_proxy_mean", 0.0))
                    <= float(teacher_metrics.get("tiny_fp_proxy_mean", 0.0)) + tiny_fp_margin
                )
            score = float(metrics.get(metric_name, 0.0))
            accepted = blend == 0.0 or (score >= teacher_score + accept_margin and volume_drift <= max_volume_drift and tiny_ok)
            record = {
                "fold_id": fold_id,
                "blend_lambda": blend,
                "accepted": accepted,
                "metrics": metrics,
                "volume_drift_fraction": volume_drift,
                "tiny_guard_ok": tiny_ok,
            }
            candidate_records.append(record)
            if accepted and (selected_record is None or score > float(selected_record["metrics"].get(metric_name, 0.0))):
                selected_record = record
        if selected_record is None:
            selected_record = candidate_records[0]
        return {
            "fold_id": fold_id,
            "blend_lambda": float(selected_record["blend_lambda"]),
            "accepted": bool(float(selected_record["blend_lambda"]) > 0.0),
            "selection_metric": metric_name,
            "teacher_metrics": teacher_metrics,
            "selected_metrics": selected_record["metrics"],
            "volume_drift_fraction": selected_record["volume_drift_fraction"],
            "candidates": candidate_records,
        }

    def _metrics_for_prediction_dict(
        self,
        cases: list[dict[str, Any]],
        predictions: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, float]:
        rows: list[dict[str, Any]] = []
        tiny_rows: list[dict[str, Any]] = []
        tiny_threshold = int(self.config.model.training.get("alignment", {}).get("tiny_lesion_voxels", 100))
        threshold = float(self.config.model.postprocessing.get("threshold", self.config.model.inference.get("threshold", 0.5)))
        for case in cases:
            case_id = case["case_id"]
            if case_id not in predictions:
                continue
            target = case["binary_target"].astype(np.uint8)
            probability = predictions[case_id]["s"].astype(np.float32)
            pred_raw = (probability >= threshold).astype(np.uint8)
            pred_post = self._postprocess_probability(probability)
            row = {
                "dice_raw": float(dice_score(pred_raw, target)),
                "dice_postprocessed": float(dice_score(pred_post, target)),
                "intersection_raw": int(np.logical_and(pred_raw.astype(bool), target.astype(bool)).sum()),
                "intersection_postprocessed": int(np.logical_and(pred_post.astype(bool), target.astype(bool)).sum()),
                "pred_positive_voxels_raw": int(pred_raw.sum()),
                "pred_positive_voxels_postprocessed": int(pred_post.sum()),
                "gt_positive_voxels": int(target.sum()),
                "fp_proxy_postprocessed": int(pred_post.sum()) - int(np.logical_and(pred_post.astype(bool), target.astype(bool)).sum()),
            }
            rows.append(row)
            if int(target.sum()) <= tiny_threshold:
                tiny_rows.append(row)
        metrics = self._aggregate_prediction_rows(rows)
        if tiny_rows:
            metrics["tiny_num_cases"] = len(tiny_rows)
            metrics["tiny_macro_dice_postprocessed"] = float(np.mean([float(row["dice_postprocessed"]) for row in tiny_rows]))
            metrics["tiny_fp_proxy_mean"] = float(np.mean([float(row["fp_proxy_postprocessed"]) for row in tiny_rows]))
        else:
            metrics["tiny_num_cases"] = 0
            metrics["tiny_macro_dice_postprocessed"] = 0.0
            metrics["tiny_fp_proxy_mean"] = 0.0
        return metrics

    def _aggregate_prediction_rows(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {
                "macro_dice_raw": 0.0,
                "macro_dice_postprocessed": 0.0,
                "micro_dice_raw": 0.0,
                "micro_dice_postprocessed": 0.0,
                "mean_pred_positive_voxels_postprocessed": 0.0,
            }

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
            "mean_pred_positive_voxels_postprocessed": float(np.mean([float(row["pred_positive_voxels_postprocessed"]) for row in rows])),
        }

    def _volume_drift_fraction(self, metrics: dict[str, float], teacher_metrics: dict[str, float]) -> float:
        selected_volume = float(metrics.get("mean_pred_positive_voxels_postprocessed", 0.0))
        teacher_volume = float(teacher_metrics.get("mean_pred_positive_voxels_postprocessed", 0.0))
        return abs(selected_volume - teacher_volume) / max(teacher_volume, 1.0)

    def status(self, round_index: int | None = None) -> str:
        rounds = self.store.list_rounds()
        if not rounds:
            raise RuntimeError("Project not initialized. Run init-project first.")
        if round_index is None:
            return self._format_status_overview(rounds)
        round_record = self.store.get_round(round_index)
        if round_record is None:
            raise RuntimeError(f"Missing round {round_index}")
        return self._format_round_status(round_record)

    def _count_selected_checkpoints(self, round_index: int) -> int:
        checkpoint_dir = self.config.workspace_root / "artifacts" / "checkpoints" / f"round_{round_index}"
        if not checkpoint_dir.exists():
            return 0
        return len([path for path in checkpoint_dir.glob("fold_*.pt") if not path.name.endswith("_last.pt")])

    def _count_revision_adapters(self, round_index: int) -> int:
        adapter_dir = self.config.workspace_root / "artifacts" / "adapters" / f"round_{round_index}"
        if not adapter_dir.exists():
            return 0
        return len([path for path in adapter_dir.glob("fold_*.pt")])

    def _round_report_flags(self, round_index: int) -> dict[str, bool]:
        report_dir = self.config.workspace_root / "reports" / f"round_{round_index}"
        return {
            "summary_json": (report_dir / "summary.json").exists(),
            "oof_summary_json": (report_dir / "oof_summary.json").exists(),
            "review_stats_csv": (report_dir / "review_stats.csv").exists(),
            "revision_oracle_csv": (report_dir / "revision_oracle_summary.csv").exists(),
            "hitl_corrected_json": (report_dir / "hitl_corrected_summary.json").exists(),
        }

    def _round_review_counts(self, round_record: dict[str, Any]) -> dict[str, int]:
        review_rows = self.store.list_review_stats(int(round_record["round_index"]))
        routine_done = sum(1 for row in review_rows if row.get("routine_final_label_path"))
        audit_anchor_done = sum(1 for row in review_rows if row.get("audit_anchor_label_path"))
        audit_final_done = sum(1 for row in review_rows if row.get("audit_final_label_path"))
        warning_count = sum(len(row.get("warnings", [])) for row in review_rows)
        routine_ids = round_record.get("routine_ids") or []
        audit_ids = round_record.get("audit_ids") or []
        return {
            "routine_total": len(routine_ids),
            "audit_total": len(audit_ids),
            "routine_done": routine_done,
            "audit_anchor_done": audit_anchor_done,
            "audit_final_done": audit_final_done,
            "review_stats_rows": len(review_rows),
            "warning_count": warning_count,
        }

    def _format_progress_summary(self, round_record: dict[str, Any]) -> str:
        counts = self._round_review_counts(round_record)
        if counts["routine_total"] == 0 and counts["audit_total"] == 0:
            return "n/a"
        return (
            f"routine:{counts['routine_done']}/{counts['routine_total']},"
            f"audit_anchor:{counts['audit_anchor_done']}/{counts['audit_total']},"
            f"audit_final:{counts['audit_final_done']}/{counts['audit_total']}"
        )

    def _format_status_overview(self, rounds: list[dict[str, Any]]) -> str:
        latest_round = max(int(row["round_index"]) for row in rounds)
        lines = [
            "Hemorrhage HITL Status",
            f"project_root={self.project_root}",
            f"workspace_root={self.config.workspace_root}",
            f"latest_round={latest_round} total_rounds={len(rounds)}",
        ]
        for round_record in rounds:
            round_idx = int(round_record["round_index"])
            budget = round_record.get("budget")
            checkpoint_count = self._count_selected_checkpoints(round_idx)
            adapter_count = self._count_revision_adapters(round_idx)
            report_flags = self._round_report_flags(round_idx)
            lines.append(
                " ".join(
                    [
                        f"round={round_idx}",
                        f"status={round_record['status']}",
                        f"budget={budget if budget is not None else '-'}",
                        f"progress={self._format_progress_summary(round_record)}",
                        f"checkpoints={checkpoint_count}/{self.config.protocol.folds}",
                        f"adapters={adapter_count}/{self.config.protocol.folds}" if adapter_count else "adapters=0",
                        "reports="
                        + ",".join(
                            [
                                f"summary={'yes' if report_flags['summary_json'] else 'no'}",
                                f"oof={'yes' if report_flags['oof_summary_json'] else 'no'}",
                                f"review_stats={'yes' if report_flags['review_stats_csv'] else 'no'}",
                            ]
                        ),
                    ]
                )
            )
        return "\n".join(lines)

    def _format_round_status(self, round_record: dict[str, Any]) -> str:
        round_idx = int(round_record["round_index"])
        progress = self._normalize_round_progress(round_record)
        counts = self._round_review_counts(round_record)
        checkpoint_count = self._count_selected_checkpoints(round_idx)
        adapter_count = self._count_revision_adapters(round_idx)
        report_flags = self._round_report_flags(round_idx)
        metrics = round_record.get("metrics", {}) or {}
        oof = metrics.get("oof", {})
        stop_state = round_record.get("stop_state", {}) or {}
        lines = [
            "Hemorrhage HITL Status",
            f"round={round_idx} status={round_record['status']} budget={round_record.get('budget') if round_record.get('budget') is not None else '-'}",
            f"review_sets: routine={counts['routine_total']} audit={counts['audit_total']}",
        ]
        if counts["routine_total"] == 0 and counts["audit_total"] == 0:
            lines.append("progress: n/a (no review sets for this round)")
        else:
            lines.append(
                "progress: "
                + " ".join(
                    [
                        f"routine_imported={'true' if progress['routine_imported'] else 'false'}",
                        f"audit_anchor_imported={'true' if progress['audit_anchor_imported'] else 'false'}",
                        f"audit_final_imported={'true' if progress['audit_final_imported'] else 'false'}",
                    ]
                )
            )
        lines.append(
            "review_counts: "
            + " ".join(
                [
                    f"routine_labels={counts['routine_done']}/{counts['routine_total']}",
                    f"audit_anchor_labels={counts['audit_anchor_done']}/{counts['audit_total']}",
                    f"audit_final_labels={counts['audit_final_done']}/{counts['audit_total']}",
                    f"review_stats_rows={counts['review_stats_rows']}",
                    f"warnings={counts['warning_count']}",
                ]
            )
        )
        lines.append(
            "artifacts: "
            + " ".join(
                [
                    f"checkpoint_dir={'yes' if checkpoint_count > 0 else 'no'}",
                    f"checkpoint_files={checkpoint_count}/{self.config.protocol.folds}",
                    f"adapter_files={adapter_count}/{self.config.protocol.folds}",
                ]
            )
        )
        lines.append(
            "reports: "
            + " ".join(
                [
                        f"summary_json={'yes' if report_flags['summary_json'] else 'no'}",
                        f"oof_summary_json={'yes' if report_flags['oof_summary_json'] else 'no'}",
                        f"review_stats_csv={'yes' if report_flags['review_stats_csv'] else 'no'}",
                        f"revision_oracle_csv={'yes' if report_flags['revision_oracle_csv'] else 'no'}",
                        f"hitl_corrected_json={'yes' if report_flags['hitl_corrected_json'] else 'no'}",
                    ]
                )
            )
        if oof:
            lines.append(
                "oof: "
                + " ".join(
                    [
                        f"macro_dice_raw={float(oof.get('macro_dice_raw', 0.0)):.6f}",
                        f"macro_dice_postprocessed={float(oof.get('macro_dice_postprocessed', 0.0)):.6f}",
                    ]
                )
            )
        else:
            lines.append("oof: n/a")
        if stop_state:
            should_stop = stop_state.get("should_stop")
            lines.append(f"stop: should_stop={str(bool(should_stop)).lower()}")
        else:
            lines.append("stop: should_stop=n/a")
        return "\n".join(lines)

    def report_round(self, round_index: int) -> None:
        logger = self._create_run_logger("report-round", round_index=round_index)
        round_record = self.store.get_round(round_index)
        if round_record is None:
            raise RuntimeError(f"Missing round {round_index}")
        report_dir = ensure_dir(self.config.workspace_root / "reports" / f"round_{round_index}")
        write_json_atomic(report_dir / "summary.json", round_record)
        with self.store.session() as conn:
            rows = conn.execute("SELECT * FROM case_metrics WHERE round_index = ? ORDER BY case_id", (round_index,)).fetchall()
        with (report_dir / "case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        review_rows = self.store.list_review_stats(round_index)
        review_csv_rows = []
        warning_rows = []
        for row in review_rows:
            warnings = list(row.get("warnings", []))
            review_csv_rows.append(
                {
                    "round_index": row["round_index"],
                    "case_id": row["case_id"],
                    "role": row["role"],
                    "routine_final_label_path": row.get("routine_final_label_path"),
                    "audit_anchor_label_path": row.get("audit_anchor_label_path"),
                    "audit_final_label_path": row.get("audit_final_label_path"),
                    "edit_ratio": row.get("edit_ratio"),
                    "whole_volume_edit_ratio": row.get("whole_volume_edit_ratio"),
                    "modified_slices_count": row.get("modified_slices_count"),
                    "anchor_assisted_dice": row.get("anchor_assisted_dice"),
                    "review_time": row.get("review_time"),
                    "anchor_time": row.get("anchor_time"),
                    "assisted_time": row.get("assisted_time"),
                    "warnings": ";".join(warnings),
                }
            )
            for warning in warnings:
                warning_rows.append(
                    {
                        "round_index": row["round_index"],
                        "case_id": row["case_id"],
                        "role": row["role"],
                        "warning": warning,
                    }
                )
        with (report_dir / "review_stats.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "round_index",
                    "case_id",
                    "role",
                    "routine_final_label_path",
                    "audit_anchor_label_path",
                    "audit_final_label_path",
                    "edit_ratio",
                    "whole_volume_edit_ratio",
                    "modified_slices_count",
                    "anchor_assisted_dice",
                    "review_time",
                    "anchor_time",
                    "assisted_time",
                    "warnings",
                ],
            )
            writer.writeheader()
            writer.writerows(review_csv_rows)
        with (report_dir / "review_warnings.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["round_index", "case_id", "role", "warning"])
            writer.writeheader()
            writer.writerows(warning_rows)
        logger.log(f"report complete | round={round_index} report_dir={report_dir}")

    def predict_external(self, model_tag: str, input_dir: Path, output_dir: Path) -> None:
        logger = self._create_run_logger("predict-external")
        del model_tag
        rounds = [row for row in self.store.list_rounds() if row["status"] == "completed"]
        if not rounds:
            raise RuntimeError("No completed rounds available")
        final_round = max(rounds, key=lambda row: row["round_index"])
        logger.log(f"predict external start | final_round={final_round['round_index']} input_dir={input_dir} output_dir={output_dir}")
        final_round_index = int(final_round["round_index"])
        revision_cfg = dict(self.config.model.training.get("revision_policy", {}))
        use_revision_policy, revision_reason = self._revision_external_available(final_round, revision_cfg)
        if use_revision_policy:
            logger.log(f"predict external using revision policy | {revision_reason}")
        elif self._round_uses_revision_policy(final_round):
            logger.log(f"predict external revision policy fallback | {revision_reason}")
            fallback_dir, fallback_source = self._resolve_revision_base_round1_checkpoint_dir(final_round_index, revision_cfg, require=False)
            checkpoint_root = fallback_dir if fallback_dir is not None else self.config.workspace_root / "artifacts" / "checkpoints" / f"round_{final_round_index}"
            logger.log(f"predict external fallback checkpoints | source={fallback_source}")
        else:
            checkpoint_root = self.config.workspace_root / "artifacts" / "checkpoints" / f"round_{final_round_index}"
        checkpoints = [checkpoint_root / f"fold_{fold_id}.pt" for fold_id in range(1, self.config.protocol.folds + 1)] if not use_revision_policy else []
        fold_selection = self._load_fold_selection(final_round_index) if not use_revision_policy else {}
        if fold_selection:
            logger.log(f"predict external using fold selection | folds={len(fold_selection)}")
        ensure_dir(output_dir)
        for image_path in sorted(Path(input_dir).glob("*_0000.nii.gz")):
            image_volume = load_nifti(image_path)
            image = image_volume.data.astype(np.float32)
            if use_revision_policy:
                prediction = self._predict_external_with_revision_policy(image, final_round_index, revision_cfg, logger)
            elif fold_selection:
                prediction = self._predict_external_with_fold_selection(image, checkpoints, fold_selection)
            else:
                prediction = self.backend.predict_external(checkpoints, image)
            np.savez_compressed(output_dir / f"{extract_case_id_from_image(image_path)}.npz", probability=prediction)
            if self._save_binary_masks_enabled():
                mask = self._postprocess_probability(prediction)
                save_nifti(output_dir / f"{extract_case_id_from_image(image_path)}.nii.gz", mask, image_volume.affine, image_volume.header, np.uint8)
            logger.log(f"predict external case complete | case_id={extract_case_id_from_image(image_path)}")
        logger.log("predict external complete")

    def _round_uses_revision_policy(self, round_record: dict[str, Any]) -> bool:
        metrics = round_record.get("metrics", {}) or {}
        revision_metrics = metrics.get("revision_policy", {}) if isinstance(metrics, dict) else {}
        return str(revision_metrics.get("mode", "")).lower() == "sgra"

    def _revision_external_available(self, round_record: dict[str, Any], revision_cfg: dict[str, Any]) -> tuple[bool, str]:
        round_index = int(round_record["round_index"])
        if not self._round_uses_revision_policy(round_record):
            return False, "final round was not produced by SGRA"
        if str(revision_cfg.get("mode", "sgra")) != "sgra":
            return False, f"revision_policy.mode is {revision_cfg.get('mode')}"
        round0_dir = self.config.workspace_root / "artifacts" / "checkpoints" / "round_0"
        missing_round0 = [fold_id for fold_id in range(1, self.config.protocol.folds + 1) if not (round0_dir / f"fold_{fold_id}.pt").exists()]
        if missing_round0:
            return False, f"missing round0 checkpoints for folds={missing_round0}"
        base_dir, base_source = self._resolve_revision_base_round1_checkpoint_dir(round_index, revision_cfg, require=False)
        if base_dir is None:
            return False, "missing base_round1 checkpoints"
        missing_base = [fold_id for fold_id in range(1, self.config.protocol.folds + 1) if not (base_dir / f"fold_{fold_id}.pt").exists()]
        if missing_base:
            return False, f"missing base_round1 checkpoints for folds={missing_base}"
        adapter_cfg = dict(revision_cfg.get("adapter", {}))
        if bool(adapter_cfg.get("enabled", True)) and bool(adapter_cfg.get("train", True)):
            adapter_dir = self.config.workspace_root / "artifacts" / "adapters" / f"round_{round_index}"
            missing_adapter = [fold_id for fold_id in range(1, self.config.protocol.folds + 1) if not (adapter_dir / f"fold_{fold_id}.pt").exists()]
            if missing_adapter:
                return False, f"missing SGRA adapters for folds={missing_adapter}"
        return True, f"base_round1_checkpoints={base_source}"

    def _predict_external_with_revision_policy(
        self,
        image: np.ndarray,
        round_index: int,
        revision_cfg: dict[str, Any],
        logger: RunLogger,
    ) -> np.ndarray:
        candidate_cfg = dict(revision_cfg.get("candidate", {}))
        selector_cfg = dict(revision_cfg.get("component_selector", {}))
        adapter_cfg = dict(revision_cfg.get("adapter", {}))
        accept_cfg = dict(revision_cfg.get("accept", {}))
        round0_dir = self.config.workspace_root / "artifacts" / "checkpoints" / "round_0"
        base_dir, _ = self._resolve_revision_base_round1_checkpoint_dir(round_index, revision_cfg, require=True)
        if base_dir is None:
            raise RuntimeError("Missing base_round1 checkpoints for revision external prediction")
        adapter_dir = self.config.workspace_root / "artifacts" / "adapters" / f"round_{round_index}"
        device = self.backend.resolve_device()
        fold_outputs: list[np.ndarray] = []
        preprocessed_image = self.backend.preprocess_image(image.astype(np.float32))
        zeros = np.zeros_like(preprocessed_image, dtype=np.uint8)
        for fold_id in range(1, self.config.protocol.folds + 1):
            p0_payload = self.backend.predict_oof_fold(round0_dir / f"fold_{fold_id}.pt", [{"case_id": "external", "image": image}])["external"]
            base_payload = self.backend.predict_oof_fold(base_dir / f"fold_{fold_id}.pt", [{"case_id": "external", "image": image}])["external"]
            p0 = p0_payload["s"].astype(np.float32)
            q0 = p0_payload.get("q", np.zeros_like(p0, dtype=np.float32)).astype(np.float32)
            p_base = base_payload["s"].astype(np.float32)
            q_base = base_payload.get("q", np.zeros_like(p_base, dtype=np.float32)).astype(np.float32)
            if not bool(revision_cfg.get("use_round0", True)):
                p0 = p_base.copy()
                q0 = np.zeros_like(q_base, dtype=np.float32)
            case = make_revision_case(
                case_id="external",
                fold_id=fold_id,
                role=None,
                image=preprocessed_image,
                p_round0=p0,
                q_round0=q0,
                p_base=p_base,
                q_base=q_base,
                y_old=zeros,
                y_final=zeros,
                candidate_cfg=candidate_cfg,
            )
            probability = case.p_base.copy()
            adapter_path = adapter_dir / f"fold_{fold_id}.pt"
            if bool(adapter_cfg.get("enabled", True)) and bool(adapter_cfg.get("train", True)) and adapter_path.exists():
                probability = predict_adapter(case, adapter_path, adapter_cfg, device=device)
            selector_path = adapter_dir / f"fold_{fold_id}_selector.json"
            if bool(selector_cfg.get("enabled", True)) and selector_path.exists():
                selector_model = json.loads(selector_path.read_text(encoding="utf-8"))
                selector_apply_cfg = dict(selector_cfg)
                if not bool(accept_cfg.get("component_no_regret", True)):
                    selector_apply_cfg["min_predicted_gain"] = 0.0
                probability, _ = apply_component_selector(case, selector_model, candidate_cfg, selector_apply_cfg, probability)
            probability, guard = apply_case_guard(
                probability,
                case.p_base,
                accept_cfg,
                threshold=float(self.config.model.postprocessing.get("threshold", self.config.model.inference.get("threshold", 0.5))),
            )
            logger.log(
                f"predict external SGRA fold complete | fold={fold_id} accepted={str(guard['accepted']).lower()} "
                f"changed_voxels={guard['changed_voxels']}"
            )
            fold_outputs.append(probability.astype(np.float32))
        return np.mean(np.stack(fold_outputs, axis=0), axis=0).astype(np.float32)

    def _load_fold_selection(self, round_index: int) -> dict[int, dict[str, Any]]:
        path = self.config.workspace_root / "reports" / f"round_{round_index}" / "fold_selection.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        output: dict[int, dict[str, Any]] = {}
        for row in payload.get("folds", []):
            fold_id = int(row.get("fold_id", 0))
            if fold_id > 0:
                output[fold_id] = row
        return output

    def _predict_external_with_fold_selection(
        self,
        image: np.ndarray,
        checkpoints: list[Path],
        fold_selection: dict[int, dict[str, Any]],
    ) -> np.ndarray:
        fold_outputs: list[np.ndarray] = []
        for fold_id, default_checkpoint in enumerate(checkpoints, start=1):
            decision = fold_selection.get(fold_id, {})
            blend_lambda = float(decision.get("blend_lambda", 1.0))
            candidate_checkpoint = Path(decision.get("candidate_checkpoint_path") or default_checkpoint)
            if blend_lambda <= 0.0:
                teacher_checkpoint_value = decision.get("teacher_checkpoint_path")
                teacher_checkpoint = Path(teacher_checkpoint_value) if teacher_checkpoint_value else None
                fold_checkpoint = teacher_checkpoint if teacher_checkpoint is not None and teacher_checkpoint.exists() else candidate_checkpoint
                if not fold_checkpoint.exists():
                    fold_checkpoint = default_checkpoint
                fold_result = self.backend.predict_oof_fold(fold_checkpoint, [{"case_id": "external", "image": image}])
                fold_outputs.append(fold_result["external"]["s"])
                continue
            if blend_lambda >= 1.0:
                fold_result = self.backend.predict_oof_fold(candidate_checkpoint, [{"case_id": "external", "image": image}])
                fold_outputs.append(fold_result["external"]["s"])
                continue
            teacher_checkpoint_value = decision.get("teacher_checkpoint_path")
            if not teacher_checkpoint_value:
                raise RuntimeError(f"Fold {fold_id} selected a teacher/candidate blend but has no teacher_checkpoint_path")
            teacher_checkpoint = Path(teacher_checkpoint_value)
            if not teacher_checkpoint.exists():
                raise RuntimeError(f"Fold {fold_id} teacher checkpoint does not exist: {teacher_checkpoint}")
            candidate_result = self.backend.predict_oof_fold(candidate_checkpoint, [{"case_id": "external", "image": image}])
            teacher_result = self.backend.predict_oof_fold(teacher_checkpoint, [{"case_id": "external", "image": image}])
            fold_prediction = (
                (1.0 - blend_lambda) * teacher_result["external"]["s"]
                + blend_lambda * candidate_result["external"]["s"]
            ).astype(np.float32)
            fold_outputs.append(fold_prediction)
        return np.mean(np.stack(fold_outputs, axis=0), axis=0).astype(np.float32)

    def normalize_review_metadata(self, input_csv: Path, output_csv: Path, required_fields: list[str]) -> None:
        normalize_review_metadata(input_csv, output_csv, required_fields)

    def _write_fold_training_log(self, report_dir: Path, fold_id: int, loss_history: list[float]) -> None:
        csv_path = report_dir / f"fold_{fold_id}_train.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            return
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
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
            for epoch, loss in enumerate(loss_history, start=1):
                writer.writerow(
                    {
                        "epoch": epoch,
                        "mean_loss": loss,
                        "num_steps": None,
                        "val_macro_dice_raw": None,
                        "val_macro_dice_postprocessed": None,
                        "val_micro_dice_raw": None,
                        "val_micro_dice_postprocessed": None,
                        "is_new_best": None,
                        "best_metric_name": None,
                        "best_metric_value": None,
                        "best_epoch": None,
                    }
                )

    def _write_fold_inference_log(self, report_dir: Path, fold_id: int, case_ids: list[str]) -> None:
        write_json_atomic(
            report_dir / f"fold_{fold_id}_inference.json",
            {
                "fold_id": fold_id,
                "num_cases": len(case_ids),
                "case_ids": case_ids,
            },
        )

    def _rewrite_fold_inference_logs(self, report_dir: Path, fold_case_ids: dict[int, list[str]], fold_rows: list[dict[str, Any]]) -> None:
        fold_metric_map = {int(row["fold_id"]): row for row in fold_rows}
        for fold_id, case_ids in fold_case_ids.items():
            payload = {
                "fold_id": fold_id,
                "num_cases": len(case_ids),
                "case_ids": case_ids,
            }
            payload.update(fold_metric_map.get(fold_id, {}))
            write_json_atomic(report_dir / f"fold_{fold_id}_inference.json", payload)

    def _compute_oof_metrics(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        threshold = float(self.config.model.postprocessing.get("threshold", self.config.model.inference.get("threshold", 0.5)))
        case_rows: list[dict[str, Any]] = []
        for case in cases:
            probability = case["previous_oof"].astype(np.float32)
            target = case["current_binary_label"].astype(np.uint8)
            pred_raw = (probability >= threshold).astype(np.uint8)
            pred_post = self._postprocess_probability(probability)
            case_rows.append(
                {
                    "case_id": case["case_id"],
                    "fold_id": int(case["fold_id"]),
                    "dice_raw": float(dice_score(pred_raw, target)),
                    "dice_postprocessed": float(dice_score(pred_post, target)),
                    "intersection_raw": int(np.logical_and(pred_raw.astype(bool), target.astype(bool)).sum()),
                    "intersection_postprocessed": int(np.logical_and(pred_post.astype(bool), target.astype(bool)).sum()),
                    "gt_positive_voxels": int(target.sum()),
                    "pred_positive_voxels_raw": int(pred_raw.sum()),
                    "pred_positive_voxels_postprocessed": int(pred_post.sum()),
                }
            )

        fold_rows = []
        for fold_id in range(1, self.config.protocol.folds + 1):
            rows = [row for row in case_rows if int(row["fold_id"]) == fold_id]
            fold_rows.append({"fold_id": fold_id, **self._aggregate_dice_rows(rows)})

        return {
            "summary": self._aggregate_dice_rows(case_rows),
            "fold_rows": fold_rows,
            "case_rows": case_rows,
        }

    def _aggregate_dice_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "num_cases": 0,
                "macro_dice_raw": 0.0,
                "macro_dice_postprocessed": 0.0,
                "micro_dice_raw": 0.0,
                "micro_dice_postprocessed": 0.0,
            }

        def _micro(pred_key: str) -> float:
            pred_sum = float(sum(int(row[pred_key]) for row in rows))
            gt_sum = float(sum(int(row["gt_positive_voxels"]) for row in rows))
            denom = pred_sum + gt_sum
            if denom == 0.0:
                return 1.0
            inter_key = "intersection_postprocessed" if pred_key.endswith("postprocessed") else "intersection_raw"
            return float((2.0 * sum(float(row[inter_key]) for row in rows)) / denom)

        return {
            "num_cases": len(rows),
            "macro_dice_raw": float(np.mean([float(row["dice_raw"]) for row in rows])),
            "macro_dice_postprocessed": float(np.mean([float(row["dice_postprocessed"]) for row in rows])),
            "micro_dice_raw": _micro("pred_positive_voxels_raw"),
            "micro_dice_postprocessed": _micro("pred_positive_voxels_postprocessed"),
        }

    def _write_oof_reports(self, report_dir: Path, oof_metrics: dict[str, Any]) -> None:
        write_json_atomic(report_dir / "oof_summary.json", oof_metrics["summary"])
        with (report_dir / "oof_case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "case_id",
                    "fold_id",
                    "dice_raw",
                    "dice_postprocessed",
                    "intersection_raw",
                    "intersection_postprocessed",
                    "gt_positive_voxels",
                    "pred_positive_voxels_raw",
                    "pred_positive_voxels_postprocessed",
                ],
            )
            writer.writeheader()
            writer.writerows(oof_metrics["case_rows"])
        with (report_dir / "oof_fold_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "fold_id",
                    "num_cases",
                    "macro_dice_raw",
                    "macro_dice_postprocessed",
                    "micro_dice_raw",
                    "micro_dice_postprocessed",
                ],
            )
            writer.writeheader()
            writer.writerows(oof_metrics["fold_rows"])

    def _save_binary_masks_enabled(self) -> bool:
        return bool(self.config.model.postprocessing.get("save_binary_masks", True))

    def _postprocess_probability(self, probability: np.ndarray) -> np.ndarray:
        post_cfg = self.config.model.postprocessing
        return postprocess_probability_map(
            probability,
            threshold=float(post_cfg.get("threshold", self.config.model.inference.get("threshold", 0.5))),
            min_component_voxels=int(post_cfg.get("min_component_voxels", 0)),
            largest_only=bool(post_cfg.get("keep_largest_component", False)),
        )

    def _save_postprocessed_mask(
        self,
        round_index: int,
        case_id: str,
        probability: np.ndarray,
        reference_path: Path,
        output_dir: Path,
    ) -> None:
        if not self._save_binary_masks_enabled():
            return
        reference = load_nifti(reference_path)
        mask = self._postprocess_probability(probability)
        mask_path = output_dir / f"{case_id}.nii.gz"
        save_nifti(mask_path, mask, reference.affine, reference.header, np.uint8)
        self.store.add_artifact(
            "postprocessed_mask",
            str(mask_path),
            sha256_file(mask_path),
            round_index=round_index,
            case_id=case_id,
            metadata=dict(self.config.model.postprocessing),
        )

    def _write_init_audit_report(self, report_dir: Path, geometry_cases: list[Any]) -> None:
        summary = summarize_case_geometries(geometry_cases)
        write_json_atomic(report_dir / "data_audit.json", summary)
        with (report_dir / "case_geometry.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "case_id",
                    "image_shape",
                    "label_shape",
                    "image_spacing",
                    "label_spacing",
                    "image_orientation",
                    "label_orientation",
                    "shape_match",
                    "spacing_match",
                    "affine_match",
                ],
            )
            writer.writeheader()
            for row in summary["cases"]:
                writer.writerow(
                    {
                        "case_id": row["case_id"],
                        "image_shape": "x".join(str(v) for v in row["image_shape"]),
                        "label_shape": "x".join(str(v) for v in row["label_shape"]),
                        "image_spacing": ",".join(str(v) for v in row["image_spacing"]),
                        "label_spacing": ",".join(str(v) for v in row["label_spacing"]),
                        "image_orientation": "".join(row["image_orientation"]),
                        "label_orientation": "".join(row["label_orientation"]),
                        "shape_match": row["shape_match"],
                        "spacing_match": row["spacing_match"],
                        "affine_match": row["affine_match"],
                    }
                )

    def _create_run_logger(self, command_name: str, round_index: int | None = None) -> RunLogger:
        log_dir = self.config.workspace_root / "logs"
        if round_index is not None:
            log_dir = log_dir / f"round_{round_index}"
        return create_run_logger(log_dir / f"{command_name}.log", prefix=command_name, mirror_stdout=True, reset=True)

    def rebase_paths(self, from_root: Path, to_root: Path) -> None:
        from_root_str = str(from_root.resolve())
        to_root_str = str(to_root.resolve())
        with self.store.session() as conn:
            for column in [
                "image_path",
                "source_label_path",
                "current_raw_label_path",
                "current_binary_label_path",
                "current_oof_path",
                "current_soft_target_path",
                "current_uncertainty_path",
            ]:
                conn.execute(
                    f"UPDATE cases SET {column} = REPLACE({column}, ?, ?) WHERE {column} LIKE ?",
                    (from_root_str, to_root_str, f"{from_root_str}%"),
                )
            conn.execute(
                "UPDATE reviews SET label_path = REPLACE(label_path, ?, ?) WHERE label_path LIKE ?",
                (from_root_str, to_root_str, f"{from_root_str}%"),
            )
            for column in [
                "routine_final_label_path",
                "audit_anchor_label_path",
                "audit_final_label_path",
            ]:
                conn.execute(
                    f"UPDATE review_stats SET {column} = REPLACE({column}, ?, ?) WHERE {column} LIKE ?",
                    (from_root_str, to_root_str, f"{from_root_str}%"),
                )
            conn.execute(
                "UPDATE artifacts SET path = REPLACE(path, ?, ?) WHERE path LIKE ?",
                (from_root_str, to_root_str, f"{from_root_str}%"),
            )
        snapshot_path = self.config.workspace_root / "project_snapshot.json"
        if snapshot_path.exists():
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            snapshot["project_root"] = to_root_str
            snapshot["workspace_root"] = str((to_root / "workspace").resolve()) if (to_root / "workspace").exists() else snapshot.get("workspace_root")
            snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
