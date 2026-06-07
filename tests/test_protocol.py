from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hemorrhage.app import build_parser
from hemorrhage.data.nifti import project_binary_mask
from hemorrhage.pipeline import Pipeline
from hemorrhage.protocol.folds import serpentine_fold_assignment
from hemorrhage.protocol.metrics import edit_ratio
from hemorrhage.protocol.rounds import build_soft_target, build_training_target_payload, compute_round_summary
from hemorrhage.protocol.selection import ScoredCase, build_audit_pool, select_audit, select_routine, split_budget
from hemorrhage.review.io import compute_review_stats
from tests.helpers import create_synthetic_project, override_project_configs


class ProtocolTests(unittest.TestCase):
    def test_binary_projection(self) -> None:
        raw = np.array([0, 1, 2, 3], dtype=np.int16)
        binary = project_binary_mask(raw)
        np.testing.assert_array_equal(binary, np.array([0, 1, 0, 1], dtype=np.uint8))

    def test_serpentine_fold_assignment(self) -> None:
        items = [(f"case_{idx}", 100 - idx) for idx in range(10)]
        assigned = serpentine_fold_assignment(items, 5)
        self.assertEqual(assigned["case_0"], 1)
        self.assertEqual(assigned["case_4"], 5)
        self.assertEqual(assigned["case_5"], 5)
        self.assertEqual(assigned["case_9"], 1)

    def test_budget_split(self) -> None:
        routine, audit = split_budget(15)
        self.assertEqual(routine, 10)
        self.assertEqual(audit, 5)

    def test_review_stats_use_union_edit_ratio_and_report_whole_volume(self) -> None:
        previous = np.zeros((4, 4, 4), dtype=np.uint8)
        current = np.zeros_like(previous)
        previous[0, 0, 0] = 1
        current[0, 0, 1] = 1
        stats = compute_review_stats(previous, current)
        self.assertAlmostEqual(stats["edit_ratio"], edit_ratio(current, previous))
        self.assertAlmostEqual(stats["edit_ratio"], 2.0 / (2.0 + 1.0e-6))
        self.assertAlmostEqual(stats["whole_volume_edit_ratio"], 2.0 / previous.size)

    def test_round_summary_uses_previous_binary_labels_for_edit_metrics(self) -> None:
        previous = np.zeros((2, 2, 2), dtype=np.uint8)
        current = np.zeros_like(previous)
        previous[0, 0, 0] = 1
        current[0, 0, 1] = 1
        case = {
            "case_id": "case_a",
            "review_count": 1,
            "previous_binary_label": current,
            "current_binary_label": current,
        }
        previous_target = previous.astype(np.float32)
        current_target = current.astype(np.float32)
        uncertainty = np.ones_like(current_target, dtype=np.float32) * 0.25
        summary = compute_round_summary(
            [case],
            routine_ids=["case_a"],
            audit_ids=[],
            previous_targets={"case_a": previous_target},
            current_targets={"case_a": current_target},
            previous_predictions={"case_a": current_target},
            current_uncertainty={"case_a": uncertainty},
            review_records={},
            previous_binary_labels={"case_a": previous},
            high_uncertainty_threshold=0.2,
        )
        self.assertGreater(summary["edit_routine"], 0.0)
        self.assertAlmostEqual(summary["delta_t"], 2.0 / previous.size)
        self.assertAlmostEqual(summary["mean_fused_uncertainty"], 0.25)
        self.assertAlmostEqual(summary["high_uncertainty_fraction"], 1.0)

    def test_alignment_target_keeps_soft_target_and_gates_contradictions(self) -> None:
        previous = np.zeros((5, 5, 5), dtype=np.uint8)
        current = np.zeros_like(previous)
        previous[2, 2, 2] = 1
        previous[3, 3, 3] = 1
        current[1, 1, 1] = 1
        current[1, 1, 2] = 1
        oof = np.zeros_like(previous, dtype=np.float32)
        oof[1, 1, 1] = 0.05
        oof[1, 1, 2] = 0.85
        oof[2, 2, 2] = 0.99
        oof[3, 3, 3] = 0.05
        teacher = np.ones_like(oof, dtype=np.float32) * 0.02
        payload = build_training_target_payload(
            previous,
            current,
            oof,
            alpha=0.15,
            reviewed=True,
            floor_weight=0.5,
            alignment_cfg={
                "enabled": True,
                "added_core_threshold": 0.3,
                "removed_core_threshold": 0.5,
                "core_dilation_iterations": 0,
                "teacher_confidence_threshold": 0.85,
            },
            teacher_probability=teacher,
        )
        expected = build_soft_target(current, oof, alpha=0.15)
        np.testing.assert_allclose(payload["target"], expected)
        self.assertEqual(payload["alignment_core_mask"][1, 1, 1], 1.0)
        self.assertEqual(payload["alignment_core_target"][1, 1, 1], 1.0)
        self.assertEqual(payload["alignment_core_mask"][2, 2, 2], 1.0)
        self.assertEqual(payload["alignment_core_target"][2, 2, 2], 0.0)
        self.assertEqual(payload["alignment_core_mask"][1, 1, 2], 0.0)
        self.assertEqual(payload["alignment_core_mask"][3, 3, 3], 0.0)
        self.assertGreater(payload["alignment_core_weight"][2, 2, 2], payload["alignment_core_weight"][1, 1, 1])
        self.assertGreater(payload["alignment_trust_weight"][0, 0, 0], 0.0)
        self.assertEqual(payload["alignment_trust_weight"][1, 1, 1], 0.0)

    def test_teacher_residual_mode_uses_teacher_target(self) -> None:
        previous = np.zeros((4, 4, 4), dtype=np.uint8)
        current = np.zeros_like(previous)
        current[1, 1, 1] = 1
        oof = np.full_like(previous, 0.05, dtype=np.float32)
        teacher = np.full_like(previous, 0.2, dtype=np.float32)
        teacher[1, 1, 1] = 0.8
        payload = build_training_target_payload(
            previous,
            current,
            oof,
            alpha=0.15,
            reviewed=True,
            alignment_cfg={
                "enabled": True,
                "mode": "teacher_residual_no_regret",
                "reviewed_supervision": "edit_only",
            },
            teacher_probability=teacher,
        )
        np.testing.assert_allclose(payload["target"], teacher, rtol=0.0, atol=1.0e-6)

    def test_teacher_residual_unreviewed_target_config_is_honored(self) -> None:
        previous = np.zeros((4, 4, 4), dtype=np.uint8)
        current = np.zeros_like(previous)
        current[1, 1, 1] = 1
        oof = np.full_like(previous, 0.1, dtype=np.float32)
        teacher = np.full_like(previous, 0.8, dtype=np.float32)
        payload = build_training_target_payload(
            previous,
            current,
            oof,
            alpha=0.15,
            reviewed=False,
            alignment_cfg={
                "enabled": True,
                "mode": "teacher_residual_no_regret",
                "unreviewed_target": "soft_label",
            },
            teacher_probability=teacher,
        )
        np.testing.assert_allclose(payload["target"], build_soft_target(current, oof, alpha=0.15))

    def test_no_regret_selector_falls_back_to_teacher(self) -> None:
        pipeline = object.__new__(Pipeline)
        pipeline.backend = SimpleNamespace(validation_metric="macro_dice_postprocessed")
        pipeline.config = SimpleNamespace(
            model=SimpleNamespace(
                postprocessing={"threshold": 0.5, "min_component_voxels": 0, "keep_largest_component": False},
                inference={"threshold": 0.5},
                training={"alignment": {"tiny_lesion_voxels": 100}},
            )
        )
        target = np.zeros((4, 4, 4), dtype=np.uint8)
        target[1:3, 1:3, 1:3] = 1
        val_cases = [{"case_id": "case_a", "binary_target": target}]
        teacher = {"case_a": {"s": target.astype(np.float32) * 0.9 + 0.05, "q": np.zeros_like(target, dtype=np.float32)}}
        candidate_prob = np.ones_like(target, dtype=np.float32) * 0.6
        candidate = {"case_a": {"s": candidate_prob, "q": np.zeros_like(target, dtype=np.float32)}}
        decision = Pipeline._select_no_regret_blend(
            pipeline,
            fold_id=1,
            val_cases=val_cases,
            teacher_predictions=teacher,
            candidate_predictions=candidate,
            alignment_cfg={
                "candidate_blend_lambdas": [0.0, 1.0],
                "accept_margin": 0.001,
                "max_volume_drift_fraction": 0.02,
                "tiny_lesion_guard": True,
            },
        )
        self.assertEqual(decision["blend_lambda"], 0.0)
        self.assertFalse(decision["accepted"])

    def test_no_regret_selector_accepts_valid_improvement(self) -> None:
        pipeline = object.__new__(Pipeline)
        pipeline.backend = SimpleNamespace(validation_metric="macro_dice_postprocessed")
        pipeline.config = SimpleNamespace(
            model=SimpleNamespace(
                postprocessing={"threshold": 0.5, "min_component_voxels": 0, "keep_largest_component": False},
                inference={"threshold": 0.5},
                training={"alignment": {"tiny_lesion_voxels": 1}},
            )
        )
        target = np.zeros((4, 4, 4), dtype=np.uint8)
        target[1:3, 1:3, 1:3] = 1
        teacher_prob = np.full_like(target, 0.05, dtype=np.float32)
        teacher_prob[1:3, 1:3, 1:2] = 0.9
        candidate_prob = target.astype(np.float32) * 0.9 + 0.05
        val_cases = [{"case_id": "case_a", "binary_target": target}]
        teacher = {"case_a": {"s": teacher_prob, "q": np.zeros_like(target, dtype=np.float32)}}
        candidate = {"case_a": {"s": candidate_prob, "q": np.zeros_like(target, dtype=np.float32)}}
        decision = Pipeline._select_no_regret_blend(
            pipeline,
            fold_id=1,
            val_cases=val_cases,
            teacher_predictions=teacher,
            candidate_predictions=candidate,
            alignment_cfg={
                "candidate_blend_lambdas": [0.0, 1.0],
                "accept_margin": 0.001,
                "max_volume_drift_fraction": 10.0,
                "tiny_lesion_guard": False,
            },
        )
        self.assertEqual(decision["blend_lambda"], 1.0)
        self.assertTrue(decision["accepted"])

    def test_routine_and_audit_selection(self) -> None:
        scored = [
            ScoredCase(case_id=f"c{idx}", fold_id=(idx % 5) + 1, review_count=idx % 3, last_review_round=idx, d=0.5, u=0.5, c=1.0, score=1.0 - idx * 0.05, score_eff=1.0 - idx * 0.05)
            for idx in range(10)
        ]
        routine = select_routine(scored, 3)
        self.assertEqual([item.case_id for item in routine], ["c0", "c1", "c2"])
        audit_pool = build_audit_pool(scored, {item.case_id for item in routine}, 2)
        audit = select_audit(audit_pool, 2, 5)
        self.assertEqual(len(audit), 2)

    def test_pipeline_migrates_old_rounds_schema(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            create_synthetic_project(tmp, num_cases=2)
            override_project_configs(tmp, project_root)
            workspace = tmp / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            db_path = workspace / "state.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE rounds (
                    round_index INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    budget INTEGER,
                    non_empty_round_index INTEGER,
                    routine_ids_json TEXT,
                    audit_ids_json TEXT,
                    config_snapshot_json TEXT,
                    metrics_json TEXT,
                    stop_state_json TEXT
                );
                """
            )
            conn.commit()
            conn.close()

            Pipeline(project_root=tmp, runtime_config_path=tmp / "configs" / "runtime.local.yaml")

            conn = sqlite3.connect(db_path)
            round_columns = {row[1] for row in conn.execute("PRAGMA table_info(rounds)").fetchall()}
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            conn.close()
            self.assertIn("progress_json", round_columns)
            self.assertIn("review_stats", tables)

    def test_status_requires_initialized_project(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            create_synthetic_project(tmp, num_cases=2)
            override_project_configs(tmp, project_root)
            pipeline = Pipeline(project_root=tmp, runtime_config_path=tmp / "configs" / "runtime.local.yaml")
            with self.assertRaisesRegex(RuntimeError, "Project not initialized. Run init-project first."):
                pipeline.status()

    def test_cli_parser_accepts_status_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status"])
        self.assertEqual(args.command, "status")
        self.assertIsNone(args.round_index)
        args = parser.parse_args(["status", "--round", "2"])
        self.assertEqual(args.command, "status")
        self.assertEqual(args.round_index, 2)
        args = parser.parse_args(["diagnose-revision-policy", "--round", "1"])
        self.assertEqual(args.command, "diagnose-revision-policy")
        self.assertEqual(args.round_index, 1)


if __name__ == "__main__":
    unittest.main()
