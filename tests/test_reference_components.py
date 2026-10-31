"""Small, deterministic checks against an explicitly provided upstream checkout."""

import importlib
import math
import os
import sys
import unittest
from pathlib import Path

SOURCE = os.environ.get("IDOL_REFERENCE_SOURCE")
if SOURCE:
    sys.path.insert(0, str(Path(SOURCE).resolve()))


@unittest.skipUnless(SOURCE, "Set IDOL_REFERENCE_SOURCE to the pinned upstream checkout")
class ReferenceComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = importlib.import_module("torch")
        cls.sample_flow = staticmethod(importlib.import_module("idol.sampler").sample_flow)
        cls.decode_camera = staticmethod(importlib.import_module("idol.camera").decode_camera)
        cls.systematic_sample = staticmethod(importlib.import_module("idol.models.decoder").systematic_sample)

    def test_sampler_guidance_matches_batched_and_separate_passes(self):
        torch = self.torch
        observed_times = []

        def velocity(sample, time, condition):
            observed_times.append(time[0].item())
            return condition

        args = dict(model=velocity, noise=torch.tensor([[10.0]]),
                    cond=torch.tensor([[2.0]]), neg_cond=torch.tensor([[1.0]]),
                    steps=2, rescale_t=3.0, cfg_strength=3.0, verbose=False)
        separate = self.sample_flow(**args, batch_cfg=False)
        self.assertEqual(observed_times, [1000.0, 1000.0, 750.0, 750.0])
        observed_times.clear()
        batched = self.sample_flow(**args, batch_cfg=True)
        self.assertEqual(observed_times, [1000.0, 750.0])
        torch.testing.assert_close(separate, torch.tensor([[5.0]]))
        torch.testing.assert_close(batched, separate)

    def test_camera_decodes_known_perspective(self):
        result = self.decode_camera(self.torch.tensor([0.0, 1.0, 0.0, 0.5, 0.25]))
        self.assertAlmostEqual(result["yaw"], 0.0)
        self.assertAlmostEqual(result["pitch"], 0.0)
        self.assertAlmostEqual(result["radius"], 2.0)
        self.assertAlmostEqual(result["fov"], math.degrees(2 * math.atan(0.125)))

    def test_octree_resampling_conserves_counts_for_one_hot(self):
        torch = self.torch
        probabilities = torch.tensor([[[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        counts = torch.tensor([[7]])
        result = self.systematic_sample(probabilities, counts)
        self.assertTrue(torch.equal(result, torch.tensor([[[0, 7, 0, 0, 0, 0, 0, 0]]])))


if __name__ == "__main__":
    unittest.main()
