"""PLY schema and value comparisons with the pinned Gaussian exporter."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class GaussianPlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import torch
        from plyfile import PlyData
        from idol_mlx.ply import save_gaussian_ply
        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.gaussian import Gaussian
        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.read_ply, cls.save, cls.gaussian = staticmethod(PlyData.read), staticmethod(save_gaussian_ply), Gaussian
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        previous = mx.default_device()
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        cls.addClassCleanup(mx.set_default_device, previous)

    def arrays(self, batch=2, count=5):
        np = self.np
        rng = np.random.default_rng(81)
        rotation = rng.standard_normal((batch, count, 4)).astype(np.float32)
        rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True)
        return {"xyz": rng.standard_normal((batch, count, 3)).astype(np.float32),
                "features_dc": rng.standard_normal((batch, count, 1, 3)).astype(np.float32),
                "scaling": rng.random((batch, count, 3), dtype=np.float32),
                "rotation": rotation, "opacity": rng.random((batch, count, 1), dtype=np.float32)}

    def test_values_schema_axis_rotation_and_clamping_match_reference(self):
        np, mx, torch = self.np, self.mx, self.torch
        data = self.arrays()
        data["opacity"][1, :3, 0] = [0, 1, 0.5]
        data["scaling"][1, 0] = [0, 1e-8, 0.5]
        data["rotation"][1, 0] = [1, 0, 0, 0]
        tensors = {name: mx.array(value) for name, value in data.items()}
        reference = self.gaussian.__new__(self.gaussian)
        for name, value in data.items():
            setattr(reference, name, torch.from_numpy(value[1].copy()))
        with tempfile.TemporaryDirectory() as directory:
            actual_path, expected_path = Path(directory) / "actual.ply", Path(directory) / "expected.ply"
            self.save(actual_path, tensors, batch_index=1)
            reference.save_ply(expected_path)
            actual, expected = self.read_ply(str(actual_path)), self.read_ply(str(expected_path))
            self.assertFalse(actual.text)
            self.assertEqual(actual.byte_order, "<")
            self.assertEqual(actual["vertex"].count, 5)
            self.assertEqual(actual["vertex"].data.dtype.names, expected["vertex"].data.dtype.names)
            for name in actual["vertex"].data.dtype.names:
                np.testing.assert_allclose(actual["vertex"][name], expected["vertex"][name],
                                           atol=2e-6, rtol=2e-6, err_msg=name)
                self.assertTrue(np.isfinite(actual["vertex"][name]).all())
            np.testing.assert_allclose([actual["vertex"][name][0] for name in ("rot_0", "rot_1", "rot_2", "rot_3")],
                                       [np.sqrt(0.5), np.sqrt(0.5), 0, 0], atol=1e-6)

    def test_binary_layout_with_known_values_without_ply_reader(self):
        import math
        import struct

        mx = self.mx
        attrs = {"xyz": mx.array([[[1.0, 2.0, 3.0]]]),
                 "features_dc": mx.array([[[[-1.0, 0.0, 1.0]]]]),
                 "scaling": mx.array([[[1.0, math.e, math.e**2]]]),
                 "rotation": mx.array([[[1.0, 0.0, 0.0, 0.0]]]),
                 "opacity": mx.array([[[0.5]]])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "known.ply"
            self.save(path, attrs)
            header, payload = path.read_bytes().split(b"end_header\n", 1)
        self.assertIn(b"format binary_little_endian 1.0\n", header)
        self.assertIn(b"element vertex 1\n", header)
        self.assertEqual(len(payload), 17 * 4)
        actual = struct.unpack("<17f", payload)
        expected = (1, -3, 2, 0, 0, 0, -1, 0, 1, 0, 0, 1, 2,
                    math.sqrt(0.5), math.sqrt(0.5), 0, 0)
        for index, (value, reference) in enumerate(zip(actual, expected)):
            with self.subTest(field=index):
                self.assertAlmostEqual(value, reference, places=6)

    def test_attribute_conversion_to_ply_matches_reference(self):
        from idol.models.decoder import GaussianFixedlenDecoder
        from idol_mlx.gaussian_attributes import decode_gaussian_attributes
        np, mx, torch = self.np, self.mx, self.torch
        config = {"num_gaussians": 2,
                  "lr": {"_xyz": 0.7, "_features_dc": 1.3, "_scaling": 0.4, "_rotation": 0.1, "_opacity": 1.2},
                  "perturbe_size": 1.5, "scaling_bias": 0.004, "opacity_bias": 0.1,
                  "3d_filter_kernel_size": 0.0009}
        ref = GaussianFixedlenDecoder(in_channels=3, model_channels=24, cond_channels=8, num_blocks=1,
                                      num_heads=2, representation_config=config)
        rng = np.random.default_rng(71)
        points = rng.random((1, 4, 3), dtype=np.float32)
        features = rng.standard_normal((1, 4, 30)).astype(np.float32)
        offsets = rng.standard_normal((2, 3)).astype(np.float32) * 0.05
        with torch.no_grad():
            ref.points_offset_perturbation.copy_(torch.from_numpy(offsets))
            ref.base_offset_scale.fill_(-0.3)
            reference = ref.to_representation(torch.from_numpy(points), torch.from_numpy(features))[0]
        attrs = decode_gaussian_attributes(mx.array(points), mx.array(features), representation_config=config,
                                           points_offset_perturbation=mx.array(offsets),
                                           base_offset_scale=mx.array(-0.3))
        with tempfile.TemporaryDirectory() as directory:
            actual_path, expected_path = Path(directory) / "actual.ply", Path(directory) / "expected.ply"
            self.save(actual_path, attrs)
            reference.save_ply(expected_path)
            actual, expected = self.read_ply(str(actual_path)), self.read_ply(str(expected_path))
            self.assertEqual(actual["vertex"].count, 8)
            for name in expected["vertex"].data.dtype.names:
                np.testing.assert_allclose(actual["vertex"][name], expected["vertex"][name],
                                           atol=2e-5, rtol=2e-4, err_msg=name)

    def test_empty_output_and_exclusive_creation(self):
        mx = self.mx
        attrs = {name: mx.array(value) for name, value in self.arrays(batch=1, count=0).items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.ply"
            self.save(path, attrs)
            self.assertEqual(self.read_ply(str(path))["vertex"].count, 0)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.save(path, attrs)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_inputs_do_not_create_files(self):
        mx = self.mx
        valid = {name: mx.array(value) for name, value in self.arrays(batch=1).items()}
        cases = [({**valid, "xyz": mx.full((1, 5, 3), float("nan"))}, 0),
                 ({**valid, "rotation": mx.full((1, 5, 4), 3e38)}, 0),
                 ({**valid, "scaling": mx.zeros((1, 4, 3))}, 0),
                 ({**valid, "opacity": valid["opacity"].astype(mx.float16)}, 0),
                 ({key: value for key, value in valid.items() if key != "rotation"}, 0),
                 (valid, 1), (valid, True)]
        with tempfile.TemporaryDirectory() as directory:
            for index, (attributes, batch_index) in enumerate(cases):
                path = Path(directory) / f"bad-{index}.ply"
                with self.subTest(index=index), self.assertRaises(ValueError):
                    self.save(path, attributes, batch_index=batch_index)
                self.assertFalse(path.exists())
            with self.assertRaisesRegex(ValueError, "end in .ply"):
                self.save(Path(directory) / "wrong.txt", valid)
