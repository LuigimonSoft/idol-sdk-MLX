"""Tests for camera output decoding and reference parity."""

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec("mlx"), "MLX is required")
class CameraTests(unittest.TestCase):
    def test_known_perspective_and_orthographic_outputs_are_json_serializable(self):
        import mlx.core as mx
        from idol_mlx.camera import decode_camera

        result = decode_camera(mx.array([0.0, 1.0, 0.0, 0.5, 0.25]))
        self.assertEqual(set(result), {"yaw", "pitch", "radius", "fov"})
        self.assertTrue(all(type(value) is float for value in result.values()))
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertAlmostEqual(result["yaw"], 0.0)
        self.assertAlmostEqual(result["pitch"], 0.0)
        self.assertAlmostEqual(result["radius"], 2.0)
        self.assertAlmostEqual(result["fov"], math.degrees(2 * math.atan(0.125)))
        fallback = decode_camera(mx.array([0.0, 0.0, 0.0, 0.0, 0.25]))
        self.assertEqual(fallback["radius"], 1000.0)
        self.assertEqual(fallback["pitch"], math.pi / 2)
        self.assertAlmostEqual(fallback["fov"], math.degrees(2 * math.atan(0.25 / 1000.0)))

    def test_shape_and_type_validation(self):
        import mlx.core as mx
        from idol_mlx.camera import decode_camera

        for invalid in (mx.zeros((4,)), mx.zeros((1, 5)), mx.zeros((6,)), [0.0] * 5):
            with self.subTest(value=type(invalid).__name__), self.assertRaisesRegex(ValueError, "shape"):
                decode_camera(invalid)


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class CameraParityTests(unittest.TestCase):
    def test_controlled_cases_match_reference(self):
        import mlx.core as mx
        import numpy as np
        import torch
        from idol_mlx.camera import decode_camera

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.camera import decode_camera as reference_decode

        previous_device = mx.default_device()
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        try:
            # Exercise both projection modes, the boundary, fallback and quadrants.
            cases = [[0., 1., 0., 0.5, 0.25], [1., -2., 3., 0.001, 0.8],
                     [-1., -1., -1., 0.0009, 1.], [0., 0., 0., 0., 0.5],
                     [1e-7, 0., 0., -0.2, 0.5], [0.999e-6, 0., 0., 0.5, 0.1],
                     [1.001e-6, 0., 0., 0.5, 0.1], [1e-5, 2e-5, -1e-5, 0.1, -0.3]]
            for dtype in ("float32", "float16"):
                for values in cases:
                    with self.subTest(dtype=dtype, values=values):
                        values = np.asarray(values, dtype=getattr(np, dtype))
                        expected = reference_decode(torch.from_numpy(values))
                        actual = decode_camera(mx.array(values))
                        self.assertEqual(actual.keys(), expected.keys())
                        for key in expected:
                            self.assertTrue(math.isclose(actual[key], expected[key], rel_tol=2e-6, abs_tol=2e-6),
                                            (key, actual[key], expected[key]))
        finally:
            mx.set_default_device(previous_device)
