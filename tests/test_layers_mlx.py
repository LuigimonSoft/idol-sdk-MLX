"""Cross-backend tests for selected upstream shared layers."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class NormalizationParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import torch
        from idol_mlx.layers import LayerNorm32, MultiHeadRMSNorm

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.models.layers import LayerNorm32 as TorchLayerNorm32, MultiHeadRMSNorm as TorchMultiHeadRMSNorm

        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.mlx_layer, cls.mlx_rms = LayerNorm32, MultiHeadRMSNorm
        cls.torch_layer, cls.torch_rms = TorchLayerNorm32, TorchMultiHeadRMSNorm
        cls.previous_device = mx.default_device()
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)

    @classmethod
    def tearDownClass(cls):
        cls.mx.set_default_device(cls.previous_device)

    def test_layer_norm_affine_and_non_affine_fp32_fp16(self):
        mx, np, torch = self.mx, self.np, self.torch
        values = np.array([[1., 2., -3., 4.], [0., 0., 0., 0.]], dtype=np.float32)
        for affine in (True, False):
            for dtype in ("float32", "float16"):
                with self.subTest(affine=affine, dtype=dtype):
                    ref = self.torch_layer(4, eps=1e-6, elementwise_affine=affine)
                    candidate = self.mlx_layer(4, eps=1e-6, elementwise_affine=affine)
                    if affine:
                        weight = np.array([0.5, -1., 2., 1.5], dtype=np.float32)
                        bias = np.array([0.1, 0.2, -0.2, 0.3], dtype=np.float32)
                        with torch.no_grad():
                            ref.weight.copy_(torch.from_numpy(weight))
                            ref.bias.copy_(torch.from_numpy(bias))
                        candidate.weight, candidate.bias = mx.array(weight), mx.array(bias)
                    x = torch.from_numpy(values).to(getattr(torch, dtype))
                    result = candidate(mx.array(values).astype(getattr(mx, dtype)))
                    mx.eval(result)
                    self.assertEqual(result.dtype, getattr(mx, dtype))
                    np.testing.assert_allclose(np.array(result).astype(np.float32),
                                               ref(x).detach().float().numpy(), rtol=1e-4 if dtype == "float32" else 1e-3,
                                               atol=1e-5 if dtype == "float32" else 1e-3)

    def test_multihead_rms_norm_fp32_fp16_and_near_zero(self):
        mx, np, torch = self.mx, self.np, self.torch
        values = np.array([[[[1., -2., 3., 4.], [0., 0., 0., 0.]],
                            [[1e-13, -2e-13, 0., 0.], [3., 2., 1., -1.]]]], dtype=np.float32)
        gamma = np.array([[0.5, 1., 1.5, -1.], [2., 0.25, -0.5, 1.]], dtype=np.float32)
        for dtype in ("float32", "float16"):
            with self.subTest(dtype=dtype):
                ref = self.torch_rms(4, 2)
                with torch.no_grad():
                    ref.gamma.copy_(torch.from_numpy(gamma))
                candidate = self.mlx_rms(4, 2)
                candidate.gamma = mx.array(gamma)
                x = torch.from_numpy(values).to(getattr(torch, dtype))
                result = candidate(mx.array(values).astype(getattr(mx, dtype)))
                mx.eval(result)
                self.assertEqual(result.dtype, getattr(mx, dtype))
                np.testing.assert_allclose(np.array(result).astype(np.float32),
                                           ref(x).detach().float().numpy(), rtol=1e-4 if dtype == "float32" else 1e-3,
                                           atol=1e-5 if dtype == "float32" else 1e-3)


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class FeedForwardParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import torch
        from idol_mlx.layers import FeedForwardNet, MLP

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.models.layers import FeedForwardNet as TorchFeedForwardNet, MLP as TorchMLP

        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.mlx_ffn, cls.mlx_mlp = FeedForwardNet, MLP
        cls.torch_ffn, cls.torch_mlp = TorchFeedForwardNet, TorchMLP
        cls.previous_device = mx.default_device()
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)

    @classmethod
    def tearDownClass(cls):
        cls.mx.set_default_device(cls.previous_device)

    def compare_models(self, reference, candidate, dtype):
        mx, np, torch = self.mx, self.np, self.torch
        reference.eval().to(getattr(torch, dtype))
        for index, layer in enumerate(reference.mlp):
            if isinstance(layer, torch.nn.Linear):
                target = candidate.mlp.layers[index]
                self.assertEqual(tuple(target.weight.shape), tuple(layer.weight.shape))
                self.assertEqual(tuple(target.bias.shape), tuple(layer.bias.shape))
                target.weight = mx.array(layer.weight.detach().float().numpy()).astype(getattr(mx, dtype))
                target.bias = mx.array(layer.bias.detach().float().numpy()).astype(getattr(mx, dtype))
        values = np.linspace(-3, 3, 24, dtype=np.float32).reshape(2, 3, 4)
        reference_input = torch.from_numpy(values).to(getattr(torch, dtype))
        actual = candidate(mx.array(values).astype(getattr(mx, dtype)))
        mx.eval(actual)
        self.assertEqual(actual.dtype, getattr(mx, dtype))
        np.testing.assert_allclose(np.array(actual).astype(np.float32),
                                   reference(reference_input).detach().float().numpy(),
                                   rtol=1e-4 if dtype == "float32" else 2e-3,
                                   atol=1e-5 if dtype == "float32" else 2e-3)

    def test_feed_forward_uses_shared_linear_weights_and_tanh_gelu(self):
        self.torch.manual_seed(17)
        for ratio in (1.5, 4.0):
            for dtype in ("float32", "float16"):
                with self.subTest(ratio=ratio, dtype=dtype):
                    self.compare_models(self.torch_ffn(4, ratio), self.mlx_ffn(4, ratio), dtype)

    def test_mlp_single_and_multiple_linear_layers(self):
        self.torch.manual_seed(23)
        for layers in (1, 2, 3):
            for dtype in ("float32", "float16"):
                with self.subTest(layers=layers, dtype=dtype):
                    channels = 7 if layers == 1 else 4
                    inner = 4 if layers == 1 else 6
                    self.compare_models(self.torch_mlp(channels, inner, 3, layers),
                                        self.mlx_mlp(channels, inner, 3, layers), dtype)


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class SinusoidalEmbedderParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import torch
        from idol_mlx.layers import SinusoidalEmbedder

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.models.layers import SinusoidalEmbedder as TorchSinusoidalEmbedder

        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.mlx_embedder, cls.torch_embedder = SinusoidalEmbedder, TorchSinusoidalEmbedder
        cls.previous_device = mx.default_device()
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)

    @classmethod
    def tearDownClass(cls):
        cls.mx.set_default_device(cls.previous_device)

    def test_temporal_embeddings_match_reference_with_identical_parameters(self):
        mx, np, torch = self.mx, self.np, self.torch
        torch.manual_seed(73)
        for size, period, two_pi in ((8, 10000, False), (10, 100, True)):
            for dtype in ("float32", "float16"):
                with self.subTest(size=size, period=period, two_pi=two_pi, dtype=dtype):
                    reference = self.torch_embedder(6, size, period, two_pi).eval()
                    candidate = self.mlx_embedder(6, size, period, two_pi)
                    for index in (0, 2):
                        source, target = reference.mlp[index], candidate.mlp.layers[index]
                        self.assertEqual(tuple(target.weight.shape), tuple(source.weight.shape))
                        self.assertEqual(tuple(target.bias.shape), tuple(source.bias.shape))
                        target.weight = mx.array(source.weight.detach().numpy())
                        target.bias = mx.array(source.bias.detach().numpy())
                    times = np.array([-1.0, 0.0, 0.5, 1.0, 3.0], dtype=np.float32)
                    reference_t = torch.from_numpy(times).to(getattr(torch, dtype))
                    actual = candidate(mx.array(times).astype(getattr(mx, dtype)))
                    mx.eval(actual)
                    expected = reference(reference_t).detach().numpy()
                    self.assertEqual(tuple(actual.shape), (5, 6))
                    self.assertEqual(actual.dtype, mx.float32)
                    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-4, atol=1e-5)


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IDOL_REFERENCE_SOURCE"),
                     "MLX and pinned reference source are required")
class SpatialEmbeddingParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import torch
        from idol_mlx.layers import PcdAbsolutePositionEmbedder, PcdAbsolutePositionEmbedderv2

        sys.path.insert(0, str(Path(os.environ["IDOL_REFERENCE_SOURCE"]).resolve()))
        from idol.models.layers import PcdAbsolutePositionEmbedder as TorchV1, PcdAbsolutePositionEmbedderv2 as TorchV2

        cls.mx, cls.np, cls.torch = mx, np, torch
        cls.mlx_v1, cls.mlx_v2 = PcdAbsolutePositionEmbedder, PcdAbsolutePositionEmbedderv2
        cls.torch_v1, cls.torch_v2 = TorchV1, TorchV2
        cls.previous_device = mx.default_device()
        device = os.environ.get("IDOL_TEST_MLX_DEVICE", "cpu")
        if device not in ("cpu", "gpu"):
            raise ValueError("IDOL_TEST_MLX_DEVICE must be cpu or gpu")
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)

    @classmethod
    def tearDownClass(cls):
        cls.mx.set_default_device(cls.previous_device)

    def test_denoiser_positions_across_padding_and_extended_frequencies(self):
        mx, np, torch = self.mx, self.np, self.torch
        values = np.array([[[0., 0.25, 1.], [0.1, 0.5, 0.9]],
                           [[1., 0.75, 0.5], [0.33, 0.66, 0.1]]], dtype=np.float32)
        for channels, max_res in ((13, 16), (30, 3)):
            for dtype in ("float32", "float16"):
                with self.subTest(channels=channels, max_res=max_res, dtype=dtype):
                    reference = self.torch_v1(channels, max_res=max_res)
                    candidate = self.mlx_v1(channels, max_res=max_res)
                    ref_input = torch.from_numpy(values).to(getattr(torch, dtype))
                    actual = candidate(mx.array(values).astype(getattr(mx, dtype)))
                    mx.eval(actual)
                    expected = reference(ref_input).detach().numpy()
                    self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                    self.assertEqual(actual.dtype, mx.float32)
                    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-4, atol=1e-5)

    def test_decoder_positions_across_padding_and_input_dtypes(self):
        mx, np, torch = self.mx, self.np, self.torch
        values = np.array([[0., 0.25, 1.], [0.125, 0.5, 0.75]], dtype=np.float32)
        for channels in (12, 13):
            for dtype in ("float32", "float16"):
                with self.subTest(channels=channels, dtype=dtype):
                    reference = self.torch_v2(channels, max_res=3)
                    candidate = self.mlx_v2(channels, max_res=3)
                    ref_input = torch.from_numpy(values).to(getattr(torch, dtype))
                    actual = candidate(mx.array(values).astype(getattr(mx, dtype)))
                    mx.eval(actual)
                    expected = reference(ref_input).detach().float().numpy()
                    self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                    self.assertEqual(actual.dtype, getattr(mx, dtype))
                    np.testing.assert_allclose(np.array(actual).astype(np.float32), expected,
                                               rtol=1e-4 if dtype == "float32" else 3e-3,
                                               atol=1e-5 if dtype == "float32" else 3e-3)
