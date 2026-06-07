from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from hemorrhage.data.nifti import load_nifti, save_nifti
from hemorrhage.pipeline import Pipeline
from hemorrhage.protocol.metrics import dice_score
from hemorrhage.protocol.rounds import build_soft_target
from hemorrhage.revision_policy import (
    adapter_forward_probability,
    apply_case_guard,
    apply_component_selector,
    compute_revision_edit_maps,
    make_revision_case,
    predict_adapter,
    train_adapter,
    train_component_selector,
)
from tests.helpers import create_synthetic_project, override_project_configs


def _base_candidate_cfg() -> dict:
    return {
        "p0_threshold": 0.3,
        "p_base_threshold": 0.5,
        "low_threshold": 0.25,
        "disagreement_threshold": 0.25,
        "teacher_uncertainty_quantile": 0.995,
        "round0_uncertainty_quantile": 0.995,
        "boundary_dilation_iterations": 1,
        "edit_dilation_iterations": 1,
        "min_component_voxels": 1,
    }


class RevisionPolicyTests(unittest.TestCase):
    def test_edit_maps_separate_review_edit_from_base_correction(self) -> None:
        y_old = np.zeros((4, 4, 4), dtype=np.uint8)
        y_final = np.zeros_like(y_old)
        p_base = np.zeros_like(y_old, dtype=np.float32)
        y_old[0, 0, 0] = 1
        y_final[1, 1, 1] = 1
        p_base[2, 2, 2] = 0.9

        edits = compute_revision_edit_maps(y_old, y_final, p_base)

        self.assertTrue(edits.review_added[1, 1, 1])
        self.assertTrue(edits.review_removed[0, 0, 0])
        self.assertTrue(edits.base_added[1, 1, 1])
        self.assertTrue(edits.base_removed[2, 2, 2])
        self.assertFalse(edits.base_removed[0, 0, 0])

    def test_adapter_probability_only_adds_base_negative_and_removes_base_positive(self) -> None:
        logits = torch.zeros((1, 4, 2, 2, 2))
        logits[:, 0] = 20.0
        logits[:, 1] = 20.0
        logits[:, 2] = 20.0
        logits[:, 3] = 20.0
        p_base = torch.full((1, 1, 2, 2, 2), 0.2)
        base_mask = torch.zeros_like(p_base)
        base_mask[:, :, 0] = 1.0
        p_base[:, :, 0] = 0.8

        p_new, *_ = adapter_forward_probability(logits, p_base, base_mask)

        self.assertTrue(torch.all(p_new[:, :, 0] < 0.01))
        self.assertTrue(torch.all(p_new[:, :, 1] > 0.99))

    def test_action_region_is_identity_outside_candidate_space(self) -> None:
        shape = (6, 6, 6)
        p0 = np.zeros(shape, dtype=np.float32)
        p_base = np.zeros(shape, dtype=np.float32)
        y = np.zeros(shape, dtype=np.uint8)
        p_base[2:4, 2:4, 2:4] = 0.9
        case = make_revision_case(
            case_id="case",
            fold_id=1,
            role=None,
            image=np.zeros(shape, dtype=np.float32),
            p_round0=p0,
            q_round0=np.zeros(shape, dtype=np.float32),
            p_base=p_base,
            q_base=np.zeros(shape, dtype=np.float32),
            y_old=y,
            y_final=y,
            candidate_cfg=_base_candidate_cfg(),
        )

        self.assertFalse(case.action_region_eval[0, 0, 0])
        candidate = p_base.copy()
        candidate[0, 0, 0] = 0.99
        candidate[~case.action_region_eval] = p_base[~case.action_region_eval]
        self.assertEqual(float(candidate[0, 0, 0]), float(p_base[0, 0, 0]))

    def test_component_selector_can_learn_round0_supported_addition(self) -> None:
        shape = (8, 8, 8)
        y = np.zeros(shape, dtype=np.uint8)
        y[3:5, 3:5, 3:5] = 1
        p0 = y.astype(np.float32) * 0.95
        p_base = np.zeros(shape, dtype=np.float32)
        case = make_revision_case(
            case_id="case",
            fold_id=1,
            role="routine",
            image=np.zeros(shape, dtype=np.float32),
            p_round0=p0,
            q_round0=np.zeros(shape, dtype=np.float32),
            p_base=p_base,
            q_base=np.zeros(shape, dtype=np.float32),
            y_old=np.zeros(shape, dtype=np.uint8),
            y_final=y,
            candidate_cfg=_base_candidate_cfg(),
        )

        selector = train_component_selector(
            [case],
            _base_candidate_cfg(),
            {"model": "logistic_ridge", "apply_threshold": 0.5, "min_predicted_gain": 0.0, "abstain_margin": 0.0},
        )
        corrected, records = apply_component_selector(
            case,
            selector,
            _base_candidate_cfg(),
            {"apply_threshold": 0.5, "min_predicted_gain": 0.0},
            p_base,
        )

        self.assertGreater(dice_score(corrected >= 0.5, y.astype(bool)), dice_score(p_base >= 0.5, y.astype(bool)))
        self.assertTrue(any(row["applied"] and row["action"] == "add" for row in records))

    def test_case_guard_falls_back_on_excessive_volume_drift(self) -> None:
        p_base = np.zeros((6, 6, 6), dtype=np.float32)
        p_base[2:4, 2:4, 2:4] = 0.9
        p_candidate = np.ones_like(p_base, dtype=np.float32) * 0.9

        guarded, meta = apply_case_guard(
            p_candidate,
            p_base,
            {
                "max_case_volume_drift_fraction": 0.05,
                "max_changed_fraction_of_base_positive": 0.10,
                "tiny_lesion_voxels": 1,
                "tiny_absolute_changed_voxels": 0,
            },
        )

        self.assertFalse(meta["accepted"])
        np.testing.assert_allclose(guarded, p_base)

    def test_adapter_training_smoke_is_finite_and_shape_preserving(self) -> None:
        shape = (8, 8, 8)
        y = np.zeros(shape, dtype=np.uint8)
        y[3:5, 3:5, 3:5] = 1
        p0 = y.astype(np.float32) * 0.95
        p_base = np.zeros(shape, dtype=np.float32)
        case = make_revision_case(
            case_id="case",
            fold_id=1,
            role="routine",
            image=np.zeros(shape, dtype=np.float32),
            p_round0=p0,
            q_round0=np.zeros(shape, dtype=np.float32),
            p_base=p_base,
            q_base=np.zeros(shape, dtype=np.float32),
            y_old=np.zeros(shape, dtype=np.uint8),
            y_final=y,
            candidate_cfg=_base_candidate_cfg(),
        )
        adapter_cfg = {
            "input_channels": ["image", "p_base", "p_round0", "abs_base_round0", "q_base", "q_round0", "boundary_distance", "old_label"],
            "epochs": 1,
            "batch_size": 1,
            "patches_per_case": 1,
            "patch_size": [8, 8, 8],
            "lr": 1.0e-4,
            "identity_loss_weight": 1.0,
            "gate_l1_weight": 0.01,
            "gate_tv_weight": 0.01,
            "volume_guard_weight": 0.01,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "adapter.pt"
            status = train_adapter([case], adapter_cfg, checkpoint, device=torch.device("cpu"), seed=7)
            self.assertTrue(checkpoint.exists())
            prediction = predict_adapter(case, checkpoint, adapter_cfg, device=torch.device("cpu"))

        self.assertEqual(prediction.shape, shape)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.isfinite(np.asarray(status["loss_history"])).all())

    def test_revision_policy_finalize_smoke_improves_synthetic_base(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            create_synthetic_project(tmp, num_cases=5, shape=(8, 8, 8))
            override_project_configs(tmp, project_root)
            model_path = tmp / "configs" / "model.yaml"
            model_cfg = yaml.safe_load(model_path.read_text(encoding="utf-8"))
            revision_cfg = model_cfg["training"]["revision_policy"]
            revision_cfg["enabled"] = True
            revision_cfg["candidate"]["min_component_voxels"] = 1
            revision_cfg["candidate"]["boundary_dilation_iterations"] = 1
            revision_cfg["candidate"]["edit_dilation_iterations"] = 1
            revision_cfg["component_selector"]["apply_threshold"] = 0.5
            revision_cfg["component_selector"]["min_predicted_gain"] = 0.0
            revision_cfg["adapter"]["epochs"] = 1
            revision_cfg["adapter"]["batch_size"] = 1
            revision_cfg["adapter"]["patches_per_case"] = 1
            revision_cfg["adapter"]["patch_size"] = [8, 8, 8]
            revision_cfg["accept"]["tiny_absolute_changed_voxels"] = 4096
            revision_cfg["accept"]["max_case_volume_drift_fraction"] = 10.0
            revision_cfg["accept"]["max_changed_fraction_of_base_positive"] = 10.0
            model_cfg["postprocessing"]["min_component_voxels"] = 1
            model_path.write_text(yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8")

            pipeline = Pipeline(project_root=tmp, runtime_config_path=tmp / "configs" / "runtime.local.yaml")
            pipeline.init_project()
            workspace = tmp / "workspace"
            round0_oof = workspace / "artifacts" / "oof" / "round_0"
            round0_soft = workspace / "artifacts" / "soft_targets" / "round_0"
            round0_unc = workspace / "artifacts" / "uncertainty" / "round_0"
            archive_oof = workspace / "archive_round1_previous_smoke" / "artifacts__oof__round_1"
            archive_unc = workspace / "archive_round1_previous_smoke" / "artifacts__uncertainty__round_1"
            routine_dir = workspace / "artifacts" / "labels" / "raw" / "round_1" / "routine"
            for directory in [round0_oof, round0_soft, round0_unc, archive_oof, archive_unc, routine_dir]:
                directory.mkdir(parents=True, exist_ok=True)

            routine_ids: list[str] = []
            for row in pipeline.store.list_cases():
                case_id = row["case_id"]
                routine_ids.append(case_id)
                label_volume = load_nifti(Path(row["current_binary_label_path"]))
                y = label_volume.data.astype(np.uint8)
                p0 = y.astype(np.float32) * 0.93 + 0.02
                p_base = np.zeros_like(p0, dtype=np.float32) + 0.02
                q = np.zeros_like(p0, dtype=np.float32)
                oof_path = round0_oof / f"{case_id}.npz"
                soft_path = round0_soft / f"{case_id}.npz"
                unc_path = round0_unc / f"{case_id}.npz"
                np.savez_compressed(oof_path, s=p0, q=q)
                np.savez_compressed(soft_path, target=build_soft_target(y, p0, pipeline.config.protocol.alpha))
                np.savez_compressed(unc_path, uncertainty=q)
                np.savez_compressed(archive_oof / f"{case_id}.npz", s=p_base, q=q)
                np.savez_compressed(archive_unc / f"{case_id}.npz", uncertainty=q)
                final_path = routine_dir / f"{case_id}.nii.gz"
                save_nifti(final_path, y, label_volume.affine, label_volume.header, np.uint8)
                row["current_oof_path"] = str(oof_path)
                row["current_soft_target_path"] = str(soft_path)
                row["current_uncertainty_path"] = str(unc_path)
                pipeline.store.upsert_case(row)
                pipeline.store.upsert_review_stats(
                    1,
                    case_id,
                    {
                        "role": "routine",
                        "routine_final_label_path": str(final_path),
                        "edit_ratio": 0.0,
                        "whole_volume_edit_ratio": 0.0,
                        "modified_slices_count": 0,
                        "review_time": None,
                        "warnings": [],
                    },
                )

            pipeline.store.upsert_round(0, {"status": "completed", "budget": None, "metrics": {"oof": {}}, "stop_state": {}})
            pipeline.store.upsert_round(
                1,
                {
                    "status": "ready_to_finalize",
                    "budget": len(routine_ids),
                    "routine_ids": routine_ids,
                    "audit_ids": [],
                    "metrics": {},
                    "stop_state": {},
                    "progress": {"routine_imported": True, "audit_anchor_imported": True, "audit_final_imported": True},
                },
            )

            pipeline.finalize_round(1)
            pipeline.report_round(1)
            completed = pipeline.store.get_round(1)
            baseline = yaml.safe_load((workspace / "reports" / "round_1" / "baseline_round1_summary.json").read_text(encoding="utf-8"))
            oof_count = len(list((workspace / "artifacts" / "oof" / "round_1").glob("*.npz")))
            adapter_count = len(list((workspace / "artifacts" / "adapters" / "round_1").glob("fold_*.pt")))
            final_macro = float(completed["metrics"]["oof"]["macro_dice_postprocessed"])
            baseline_macro = float(baseline["macro_dice_postprocessed"])

            self.assertIsNotNone(completed)
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(oof_count, 5)
            self.assertEqual(adapter_count, 5)
            self.assertGreater(final_macro, baseline_macro)


if __name__ == "__main__":
    unittest.main()
