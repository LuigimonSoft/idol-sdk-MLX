"""Synthetic end-to-end denoiser forward parity against the upstream SDK."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class DenoiserParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import torch
        from mlx.utils import tree_flatten
        from tensordict import TensorDict
        from idol_mlx.dit import LatentSeqMMFlowModel

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.models.dit import LatentSeqMMFlowModel as TorchDenoiser

        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.tree_flatten, cls.tensor_dict = staticmethod(tree_flatten), TensorDict
        cls.mlx_model, cls.torch_model = LatentSeqMMFlowModel, TorchDenoiser
        cls.previous_device = mx.default_device()
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)

    @classmethod
    def tearDownClass(cls):
        cls.mx.set_default_device(cls.previous_device)

    def test_tiny_fp32_forward_with_all_shared_synthetic_weights(self):
        mx, np, torch = self.mx, self.np, self.torch
        torch.manual_seed(103)
        config = dict(q_token_length=3, in_channels=4, cam_channels=5, model_channels=24,
                      cond_channels=6, cond2_channels=7, num_blocks=1, num_refiner_blocks=2,
                      num_heads=2, num_head_channels=12, mlp_ratio=1.5)
        ref = self.torch_model(**config).eval().float()
        ref.dtype = torch.float32
        for collection in (ref.noise_repo_layers, ref.context_repo_layers, ref.repo_layers):
            for layer in collection:
                layer.real_valued = True
        with torch.no_grad():
            ref.pos_pe.copy_(torch.tensor([[[0.1, 0.3, 0.5], [0.7, 0.2, 0.4], [0.9, 0.6, 0.8]]]))
        model = self.mlx_model(**config)
        target = dict(self.tree_flatten(model.parameters()))
        mapped = {}
        for name, tensor in ref.state_dict().items():
            mlx_name = (name.replace("adaLN_modulation.1.", "adaLN_modulation.layers.1.")
                        .replace("t_embedder.mlp.", "t_embedder.mlp.layers.")
                        .replace("cam_refiner.mlp.", "cam_refiner.mlp.layers.")
                        .replace(".mlp.mlp.", ".mlp.mlp.layers."))
            self.assertIn(mlx_name, target)
            self.assertEqual(tuple(target[mlx_name].shape), tuple(tensor.shape))
            mapped[mlx_name] = mx.array(tensor.detach().numpy())
        self.assertEqual(set(mapped), set(target))
        model.load_weights(list(mapped.items()), strict=True)
        rng = np.random.default_rng(113)
        for batch, cond_len in ((1, 2), (2, 4)):
            with self.subTest(batch=batch, cond_len=cond_len):
                latent = rng.normal(size=(batch, 3, 4)).astype(np.float32)
                camera = rng.normal(size=(batch, 1, 5)).astype(np.float32)
                feature1 = rng.normal(size=(batch, cond_len, 6)).astype(np.float32)
                feature2 = rng.normal(size=(batch, cond_len, 7)).astype(np.float32)
                times = np.linspace(50, 900, batch, dtype=np.float32)
                x_t = self.tensor_dict({"latent": torch.from_numpy(latent), "camera": torch.from_numpy(camera)},
                                       batch_size=[batch])
                cond_t = self.tensor_dict({"feature1": torch.from_numpy(feature1), "feature2": torch.from_numpy(feature2)},
                                          batch_size=[batch])
                expected = ref(x_t, torch.from_numpy(times), cond_t)
                actual = model({"latent": mx.array(latent), "camera": mx.array(camera)}, mx.array(times),
                               {"feature1": mx.array(feature1), "feature2": mx.array(feature2)})
                mx.eval(actual)
                for key, shape in (("latent", latent.shape), ("camera", camera.shape)):
                    self.assertEqual(tuple(actual[key].shape), shape)
                    self.assertEqual(actual[key].dtype, mx.float32)
                    np.testing.assert_allclose(np.array(actual[key]), expected[key].detach().numpy(),
                                               rtol=2e-4, atol=2e-5)
                zero_cond_t = self.tensor_dict({"feature1": torch.zeros_like(cond_t["feature1"]),
                                                "feature2": torch.zeros_like(cond_t["feature2"])}, batch_size=[batch])
                zero_expected = ref(x_t, torch.from_numpy(times), zero_cond_t)["latent"].detach().numpy()
                self.assertGreater(float(np.max(np.abs(expected["latent"].detach().numpy() - zero_expected))), 1e-5)
                zero_actual = model({"latent": mx.array(latent), "camera": mx.array(camera)}, mx.array(times),
                                    {"feature1": mx.zeros_like(mx.array(feature1)),
                                     "feature2": mx.zeros_like(mx.array(feature2))})
                mx.eval(zero_actual)
                np.testing.assert_allclose(np.array(zero_actual["latent"]), zero_expected, rtol=2e-4, atol=2e-5)
