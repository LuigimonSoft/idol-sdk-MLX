"""Check synthetic fixture behavior when reference dependencies are available."""

import importlib
import os
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("IDOL_REFERENCE_SOURCE"), "Reference source not configured")
class CaptureReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        cls.capture_module = importlib.import_module("scripts.capture_reference")
        cls.torch = importlib.import_module("torch")
        cls.load_file = staticmethod(importlib.import_module("safetensors.torch").load_file)

    def test_round_trip_and_known_sampler_camera_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reference.safetensors"
            self.capture_module.write_capture(path)
            values = self.load_file(str(path))
            self.assertEqual(tuple(values["repo.input"].shape), (1, 3, 24))
            self.assertEqual(tuple(values["repo.output"].shape), (1, 3, 2, 6, 2))
            self.assertEqual(tuple(values["attention.output"].shape), (1, 3, 24))
            self.torch.testing.assert_close(values["sampler.output"], self.torch.tensor([[5.0]]))
            self.torch.testing.assert_close(values["camera.output"][:3], self.torch.tensor([0., 0., 2.], dtype=self.torch.float64))
            self.assertIn("repo.weights.final_map.weight", values)
            self.assertIn("attention.weights.qkv.weight", values)
            with self.assertRaises(FileExistsError):
                self.capture_module.write_capture(path)

    def test_captured_weights_reproduce_component_outputs(self):
        torch = self.torch
        tensors = self.capture_module.capture()
        layers = importlib.import_module("idol.models.layers")
        repo = layers.RePo3DRotaryEmbedding(24, num_heads=2, head_dim=12)
        repo.real_valued = True
        attention = layers.RopeMultiHeadAttention(24, num_heads=2)
        for prefix, module in (("repo", repo), ("attention", attention)):
            key_prefix = f"{prefix}.weights."
            module.load_state_dict({key[len(key_prefix):]: value for key, value in tensors.items()
                                    if key.startswith(key_prefix)}, strict=True)
        with torch.no_grad():
            rotary = repo(tensors["repo.input"])
            output = attention(tensors["attention.input"], rotary)
        torch.testing.assert_close(rotary, tensors["repo.output"])
        torch.testing.assert_close(output, tensors["attention.output"])


if __name__ == "__main__":
    unittest.main()
