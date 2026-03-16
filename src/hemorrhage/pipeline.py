"""High-level orchestration for the protocol-first hemorrhage pipeline."""

from __future__ import annotations

import csv
import json
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
    build_voxel_weights,
    compute_review_reentry,
    compute_round_summary,
    compute_stop_state,
    compute_uncertainty_from_target,
)
from hemorrhage.protocol.selection import build_audit_pool, compute_case_scores, select_audit, select_routine, split_budget
from hemorrhage.review.io import compute_review_stats, export_review_bundle, import_review_label, normalize_review_metadata, validate_import_dir
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

        for case in previous_cases:
            case_id = case["case_id"]
            previous_targets[case_id] = np.load(case["current_soft_target_path"])["target"].astype(np.float32)
            previous_predictions[case_id] = case["previous_oof"].astype(np.float32)
            current_raw = load_nifti(Path(case["current_raw_label_path"]))
            current_binary = case["current_binary_label"].astype(np.uint8)
            role = None
            if case_id in round_record["routine_ids"]:
                role = "routine"
                stats_row = review_stats_map.get(case_id)
                if stats_row is None or not stats_row.get("routine_final_label_path"):
                    raise RuntimeError(f"Missing routine review stats for {case_id}")
                imported = import_review_label(Path(stats_row["routine_final_label_path"]))
                current_raw = load_nifti(Path(stats_row["routine_final_label_path"]))
                current_binary = imported["binary"]
                review_records[case_id] = {**stats_row, "final_binary": current_binary}
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
            soft_target = build_soft_target(current_binary, case["previous_oof"], self.config.protocol.alpha)
            current_targets[case_id] = soft_target
            train_unc = compute_uncertainty_from_target(soft_target, self.config.protocol.alpha)
            prepared_cases.append(
                {
                    "case_id": case_id,
                    "fold_id": int(case["fold_id"]),
                    "image": case["image"],
                    "target": soft_target,
                    "binary_target": current_binary.astype(np.uint8),
                    "voxel_weight": build_voxel_weights(train_unc, reviewed=role is not None, floor_weight=float(self.config.protocol.loss["voxel_weight_floor"])),
                    "case_weight": float(self.config.protocol.loss["case_weight_reviewed"] if role is not None else self.config.protocol.loss["case_weight_unreviewed"]),
                }
            )

        checkpoint_dir = ensure_dir(self.config.workspace_root / "artifacts" / "checkpoints" / f"round_{round_index}")
        report_dir = ensure_dir(self.config.workspace_root / "reports" / f"round_{round_index}")
        round_metrics_history = []
        fold_case_ids: dict[int, list[str]] = {}
        for existing_round in range(1, round_index):
            existing = self.store.get_round(existing_round)
            if existing is not None and existing["metrics"]:
                round_metrics_history.append(existing["metrics"])

        for fold_id in range(1, self.config.protocol.folds + 1):
            train_cases = [case for case in prepared_cases if case["fold_id"] != fold_id]
            holdout_cases = [{"case_id": case["case_id"], "image": case["image"]} for case in prepared_cases if case["fold_id"] == fold_id]
            fold_case_ids[fold_id] = [item["case_id"] for item in holdout_cases]
            previous_ckpt = self.config.workspace_root / "artifacts" / "checkpoints" / f"round_{round_index - 1}" / f"fold_{fold_id}.pt"
            fold_logger = logger.child(f"fold={fold_id}")
            fold_logger.log(
                f"fold start | train_cases={len(train_cases)} holdout_cases={len(holdout_cases)} "
                f"resume={'yes' if previous_ckpt.exists() else 'no'}"
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
        checkpoint_root = self.config.workspace_root / "artifacts" / "checkpoints" / f"round_{final_round['round_index']}"
        checkpoints = [checkpoint_root / f"fold_{fold_id}.pt" for fold_id in range(1, self.config.protocol.folds + 1)]
        ensure_dir(output_dir)
        for image_path in sorted(Path(input_dir).glob("*_0000.nii.gz")):
            image_volume = load_nifti(image_path)
            prediction = self.backend.predict_external(checkpoints, image_volume.data.astype(np.float32))
            np.savez_compressed(output_dir / f"{extract_case_id_from_image(image_path)}.npz", probability=prediction)
            if self._save_binary_masks_enabled():
                mask = self._postprocess_probability(prediction)
                save_nifti(output_dir / f"{extract_case_id_from_image(image_path)}.nii.gz", mask, image_volume.affine, image_volume.header, np.uint8)
            logger.log(f"predict external case complete | case_id={extract_case_id_from_image(image_path)}")
        logger.log("predict external complete")

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
