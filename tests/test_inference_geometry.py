import unittest

import nibabel as nib
import numpy as np

from fastsurfer_vinn.inference import _in_plane_voxel_size


class InferenceGeometryTest(unittest.TestCase):
    def test_spacing_is_read_in_ras_order(self):
        affine = np.array([[0, 0, 0.8, 0], [0, -0.8, 0, 0], [2, 0, 0, 0], [0, 0, 0, 1]])
        img = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), affine)
        self.assertAlmostEqual(_in_plane_voxel_size(img, None), 0.8)

    def test_unequal_spacing_requires_an_explicit_scale(self):
        img = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.diag([0.8, 1, 2, 1]))
        with self.assertRaisesRegex(ValueError, "in-plane spacings differ"):
            _in_plane_voxel_size(img, None)
        self.assertEqual(_in_plane_voxel_size(img, 0.9), 0.9)
