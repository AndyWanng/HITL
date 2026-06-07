from __future__ import annotations

import shutil
import tempfile
import unittest
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from hemorrhage.pipeline import Pipeline
from tests.helpers import create_synthetic_project, override_project_configs


@unittest.skipUnless(torch.cuda.is_available(), "CUDA smoke test requires a local GPU")
class SmokeTests(unittest.TestCase):
    def test_full_round_smoke(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            create_synthetic_project(tmp, num_cases=10)
            override_project_configs(tmp, project_root)
            pipeline = Pipeline(project_root=tmp, runtime_config_path=tmp / "configs" / "runtime.local.yaml")
            pipeline.init_project()
            self.assertTrue((tmp / "workspace" / "reports" / "init" / "data_audit.json").exists())
            pipeline.train_round0()
            round0_status = pipeline.status()
            self.assertIn("round=0 status=completed", round0_status)
            self.assertIn("progress=n/a", round0_status)
            round0_detail = pipeline.status(round_index=0)
            self.assertIn("round=0 status=completed", round0_detail)
            self.assertIn("checkpoint_files=5/5", round0_detail)
            self.assertIn("oof: macro_dice_raw=", round0_detail)
            pipeline.plan_round(round_index=1, budget=5)
            planned_status = pipeline.status(round_index=1)
            self.assertIn("round=1 status=awaiting_inputs", planned_status)
            self.assertIn("routine_imported=false", planned_status)
            self.assertIn("audit_anchor_imported=false", planned_status)
            self.assertIn("audit_final_imported=false", planned_status)

            import csv

            routine_manifest = tmp / "workspace" / "review" / "round_1" / "routine" / "manifest.csv"
            audit_anchor_manifest = tmp / "workspace" / "review" / "round_1" / "audit_anchor" / "manifest.csv"
            audit_final_manifest = tmp / "workspace" / "review" / "round_1" / "audit_final" / "manifest.csv"
            self.assertTrue(audit_final_manifest.exists())
            with audit_final_manifest.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    self.assertTrue(Path(row["model_mask_path"]).exists())
                    self.assertTrue(Path(row["uncertainty_path"]).exists())
                    self.assertEqual(row["seed_source"], "current_label_pending_anchor_update")

            routine_out = tmp / "routine_out"
            (routine_out / "labels").mkdir(parents=True, exist_ok=True)
            rows = []
            with routine_manifest.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    case_id = row["case_id"]
                    self.assertTrue(Path(row["model_mask_path"]).exists())
                    self.assertTrue(Path(row["uncertainty_path"]).exists())
                    shutil.copy2(row["seed_label_path"], routine_out / "labels" / f"{case_id}.nii.gz")
                    rows.append({"case_id": case_id, "review_time": "12.0"})
            with (routine_out / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
                writer.writeheader()
                writer.writerows(rows)
            pipeline.import_routine(1, routine_out)
            routine_imported_status = pipeline.status(round_index=1)
            self.assertIn("round=1 status=awaiting_inputs", routine_imported_status)
            self.assertIn("routine_imported=true", routine_imported_status)
            self.assertIn("audit_anchor_imported=false", routine_imported_status)
            self.assertIn("audit_final_imported=false", routine_imported_status)

            audit_anchor_out = tmp / "audit_anchor_out"
            (audit_anchor_out / "labels").mkdir(parents=True, exist_ok=True)
            rows = []
            anchor_arrays: dict[str, np.ndarray] = {}
            with audit_anchor_manifest.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    case_id = row["case_id"]
                    self.assertNotIn("model_mask_path", row)
                    self.assertNotIn("uncertainty_path", row)
                    label = nib.load(row["seed_label_path"])
                    data = label.get_fdata().astype("int16").copy()
                    positives = np.argwhere((data == 1) | (data == 3))
                    if positives.size:
                        x, y, z = positives[0]
                        data[x, y, z] = 3 if data[x, y, z] == 1 else 1
                    else:
                        data[0, 0, 0] = 1
                    anchor_arrays[case_id] = data
                    nib.save(nib.Nifti1Image(data, label.affine, header=label.header), str(audit_anchor_out / "labels" / f"{case_id}.nii.gz"))
                    rows.append({"case_id": case_id})
            with (audit_anchor_out / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case_id"])
                writer.writeheader()
                writer.writerows(rows)
            pipeline.import_audit_anchor(1, audit_anchor_out)
            anchor_imported_status = pipeline.status(round_index=1)
            self.assertIn("round=1 status=awaiting_audit_final", anchor_imported_status)
            self.assertIn("routine_imported=true", anchor_imported_status)
            self.assertIn("audit_anchor_imported=true", anchor_imported_status)
            self.assertIn("audit_final_imported=false", anchor_imported_status)
            with audit_final_manifest.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    case_id = row["case_id"]
                    self.assertEqual(row["seed_source"], "anchor_label")
                    refreshed = nib.load(row["seed_label_path"]).get_fdata().astype("int16")
                    np.testing.assert_array_equal(refreshed, anchor_arrays[case_id])

            audit_final_out = tmp / "audit_final_out"
            (audit_final_out / "labels").mkdir(parents=True, exist_ok=True)
            rows = []
            with audit_final_manifest.open("r", encoding="utf-8", newline="") as handle:
                for index, row in enumerate(csv.DictReader(handle)):
                    case_id = row["case_id"]
                    self.assertTrue(Path(row["model_mask_path"]).exists())
                    self.assertTrue(Path(row["uncertainty_path"]).exists())
                    label = nib.load(row["seed_label_path"])
                    data = label.get_fdata().astype("int16")
                    data = data.copy()
                    data[data == 3] = 1
                    nib.save(nib.Nifti1Image(data, label.affine, header=label.header), str(audit_final_out / "labels" / f"{case_id}.nii.gz"))
                    meta = {"case_id": case_id}
                    if index == 0:
                        meta["assisted_time"] = "6.0"
                    rows.append(meta)
            with (audit_final_out / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["assisted_time", "case_id"])
                writer.writeheader()
                writer.writerows(rows)
            pipeline.import_audit_final(1, audit_final_out)
            ready_status = pipeline.status(round_index=1)
            self.assertIn("round=1 status=ready_to_finalize", ready_status)
            self.assertIn("routine_imported=true", ready_status)
            self.assertIn("audit_anchor_imported=true", ready_status)
            self.assertIn("audit_final_imported=true", ready_status)
            pipeline.finalize_round(1)
            pipeline.report_round(1)
            completed_status = pipeline.status(round_index=1)
            self.assertIn("round=1 status=completed", completed_status)
            self.assertIn("checkpoint_files=5/5", completed_status)
            self.assertIn("summary_json=yes", completed_status)
            self.assertIn("review_stats_csv=yes", completed_status)
            self.assertIn("stop: should_stop=", completed_status)

            self.assertTrue((tmp / "workspace" / "reports" / "round_1" / "summary.json").exists())
            self.assertTrue((tmp / "workspace" / "artifacts" / "checkpoints" / "round_1" / "fold_1.pt").exists())
            self.assertTrue((tmp / "workspace" / "artifacts" / "masks" / "round_1").exists())
            self.assertTrue((tmp / "workspace" / "reports" / "round_1" / "oof_summary.json").exists())
            self.assertTrue((tmp / "workspace" / "reports" / "round_1" / "oof_fold_metrics.csv").exists())
            self.assertTrue((tmp / "workspace" / "reports" / "round_1" / "review_stats.csv").exists())
            self.assertTrue((tmp / "workspace" / "reports" / "round_1" / "review_warnings.csv").exists())
            self.assertTrue((tmp / "workspace" / "logs" / "round_0" / "train-round0.log").exists())
            self.assertTrue((tmp / "workspace" / "logs" / "round_1" / "finalize-round.log").exists())
            self.assertTrue((tmp / "workspace" / "reports" / "round_1" / "fold_1_train_status.json").exists())
            self.assertTrue((tmp / "workspace" / "reports" / "round_1" / "fold_1_inference_status.json").exists())
            fold_selection_path = tmp / "workspace" / "reports" / "round_1" / "fold_selection.json"
            self.assertTrue(fold_selection_path.exists())
            fold_selection = json.loads(fold_selection_path.read_text(encoding="utf-8"))
            self.assertEqual(len(fold_selection["folds"]), 5)
            for decision in fold_selection["folds"]:
                self.assertIn("blend_lambda", decision)
                self.assertIn("teacher_metrics", decision)
                self.assertIn("selected_metrics", decision)
                self.assertIn("teacher_checkpoint_path", decision)
            summary = json.loads((tmp / "workspace" / "reports" / "round_1" / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("oof", summary["metrics"])
            self.assertIn("macro_dice_raw", summary["metrics"]["oof"])
            with (tmp / "workspace" / "reports" / "round_1" / "fold_1_train.csv").open("r", encoding="utf-8", newline="") as handle:
                train_rows = list(csv.DictReader(handle))
            self.assertTrue(train_rows)
            self.assertIn("val_macro_dice_postprocessed", train_rows[0])
            train_status = json.loads((tmp / "workspace" / "reports" / "round_1" / "fold_1_train_status.json").read_text(encoding="utf-8"))
            self.assertIn("best_metric_name", train_status)
            self.assertIn("best_metric_value", train_status)
            with (tmp / "workspace" / "reports" / "round_1" / "review_warnings.csv").open("r", encoding="utf-8", newline="") as handle:
                warning_rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["warning"] == "missing_anchor_time" for row in warning_rows))
            self.assertTrue(any(row["warning"] == "missing_assisted_time" for row in warning_rows))

            external_input = tmp / "external_input"
            external_output = tmp / "external_output"
            external_input.mkdir()
            source_image = next((tmp / "data" / "imagesTr").glob("*_0000.nii.gz"))
            shutil.copy2(source_image, external_input / source_image.name)
            pipeline.predict_external("latest", external_input, external_output)
            predicted_npz = external_output / source_image.name.replace("_0000.nii.gz", ".npz")
            predicted_mask = external_output / source_image.name.replace("_0000.nii.gz", ".nii.gz")
            self.assertTrue(predicted_npz.exists())
            self.assertTrue(predicted_mask.exists())
            prediction = np.load(predicted_npz)["probability"]
            self.assertEqual(prediction.shape, nib.load(str(source_image)).shape)
            self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
