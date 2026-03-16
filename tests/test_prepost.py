from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from hemorrhage.data.nifti import inspect_case_geometry, validate_case_geometry
from hemorrhage.training.postprocess import postprocess_probability_map


class GeometryAuditTests(unittest.TestCase):
    def test_geometry_validation_accepts_matching_image_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "case_0000.nii.gz"
            label_path = tmp / "case.nii.gz"
            affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
            nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), affine), str(image_path))
            nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.int16), affine), str(label_path))
            geometry = inspect_case_geometry("case", image_path, label_path, spacing_tolerance=1.0e-4, affine_tolerance=1.0e-4)
            validate_case_geometry(geometry)

    def test_geometry_validation_rejects_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "case_0000.nii.gz"
            label_path = tmp / "case.nii.gz"
            affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
            nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), affine), str(image_path))
            nib.save(nib.Nifti1Image(np.zeros((8, 8, 7), dtype=np.int16), affine), str(label_path))
            geometry = inspect_case_geometry("case", image_path, label_path, spacing_tolerance=1.0e-4, affine_tolerance=1.0e-4)
            with self.assertRaises(ValueError):
                validate_case_geometry(geometry)


class PostprocessingTests(unittest.TestCase):
    def test_postprocess_removes_small_components(self) -> None:
        probability = np.zeros((12, 12, 12), dtype=np.float32)
        probability[1:5, 1:5, 1:5] = 0.8
        probability[9:11, 9:11, 9:11] = 0.9
        mask = postprocess_probability_map(probability, threshold=0.5, min_component_voxels=16, largest_only=False)
        self.assertEqual(int(mask.sum()), 64)

    def test_postprocess_can_keep_only_largest_component(self) -> None:
        probability = np.zeros((12, 12, 12), dtype=np.float32)
        probability[1:5, 1:5, 1:5] = 0.8
        probability[6:10, 6:10, 6:10] = 0.8
        mask = postprocess_probability_map(probability, threshold=0.5, min_component_voxels=0, largest_only=True)
        self.assertEqual(int(mask.sum()), 64)


if __name__ == "__main__":
    unittest.main()
