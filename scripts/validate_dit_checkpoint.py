"""Compare a reduced FP32 denoiser forward pass with a trusted checkpoint."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
import torch
from safetensors.torch import load_file
from tensordict import TensorDict

from idol.models.dit import LatentSeqMMFlowModel as TorchModel
from idol.models.layers import PcdAbsolutePositionEmbedder as TorchPositionEmbedder
from idol_mlx.config import load_config
from idol_mlx.dit import LatentSeqMMFlowModel
from idol_mlx.dit_weights import load_dit_checkpoint
from scripts.parity_metrics import compare


def compare_checkpoint(config_path, checkpoint_path, *, tokens=8, context_tokens=4,
                       atol=2e-5, rtol=2e-4):
    """Return metrics for fixed synthetic inputs with a shortened latent sequence.

    The complete checkpoint is loaded strictly before cropping positional tokens.
    Only local trusted inputs are accepted; neither checkpoint nor generated
    inputs are written or published. The PyTorch reference is explicitly FP32.
    """
    config = load_config(config_path)["denoiser"]["args"]
    if (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1 or
            tokens > config["q_token_length"]):
        raise ValueError("tokens must be between 1 and q_token_length.")
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int) or context_tokens < 1:
        raise ValueError("context_tokens must be a positive integer.")
    # Meta construction avoids allocating random full-size reference parameters.
    with torch.device("meta"):
        reference = TorchModel(**config)
    reference.load_state_dict(load_file(str(checkpoint_path)), strict=True, assign=True)
    # This upstream frequency tensor is not a persistent checkpoint buffer.
    reference.pos_embedder = TorchPositionEmbedder(config["model_channels"])
    reference.eval()
    reference.dtype = torch.float32
    for collection in (reference.noise_repo_layers, reference.context_repo_layers, reference.repo_layers):
        for layer in collection:
            layer.real_valued = True
    candidate = LatentSeqMMFlowModel(**config)
    load_dit_checkpoint(candidate, checkpoint_path)
    reference.pos_pe = reference.pos_pe[:, :tokens, :]
    candidate.pos_pe = candidate.pos_pe[:, :tokens, :]
    generator = np.random.default_rng(2026)
    latent = generator.standard_normal((1, tokens, config["in_channels"])).astype(np.float32)
    camera = generator.standard_normal((1, 1, config["cam_channels"])).astype(np.float32)
    feature1 = generator.standard_normal((1, context_tokens, config["cond_channels"])).astype(np.float32)
    feature2 = generator.standard_normal((1, context_tokens, config["cond2_channels"])).astype(np.float32)
    time = np.array([500.0], dtype=np.float32)
    x_t = TensorDict({"latent": torch.from_numpy(latent), "camera": torch.from_numpy(camera)}, batch_size=[1])
    cond_t = TensorDict({"feature1": torch.from_numpy(feature1), "feature2": torch.from_numpy(feature2)},
                        batch_size=[1])
    with torch.no_grad():
        expected_position = reference.pos_embedder(reference.pos_pe).numpy()
        expected = reference(x_t, torch.from_numpy(time), cond_t)
    actual_position = candidate.pos_embedder(candidate.pos_pe)
    # Isolate frequency generation by using the reference frequencies in MLX.
    reference_frequencies = mx.array(reference.pos_embedder.freqs.numpy())
    angles = mx.outer(candidate.pos_pe.reshape(-1), reference_frequencies) * 2 * math.pi
    shared_position = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1).reshape(1, tokens, -1)
    padding = config["model_channels"] - shared_position.shape[-1]
    if padding > 0:
        shared_position = mx.concatenate([shared_position, mx.zeros((1, tokens, padding))], axis=-1)
    actual = candidate({"latent": mx.array(latent), "camera": mx.array(camera)}, mx.array(time),
                       {"feature1": mx.array(feature1), "feature2": mx.array(feature2)})
    mx.eval(actual_position, shared_position, actual)
    # Reuse the same embedding only for this diagnostic forward pass.
    # This is not a production frequency-generation strategy.
    candidate.pos_embedder = lambda _positions: shared_position
    shared_actual = candidate({"latent": mx.array(latent), "camera": mx.array(camera)}, mx.array(time),
                              {"feature1": mx.array(feature1), "feature2": mx.array(feature2)})
    mx.eval(shared_actual)
    comparisons = {
        "position": asdict(compare(expected_position, np.array(actual_position), atol=atol, rtol=rtol)),
        "position_shared_frequencies": asdict(compare(expected_position, np.array(shared_position),
                                                       atol=atol, rtol=rtol)),
    }
    for name in ("latent", "camera"):
        comparisons[name] = asdict(compare(expected[name].detach().numpy(), np.array(actual[name]),
                                           atol=atol, rtol=rtol))
        comparisons[f"{name}_shared_frequencies"] = asdict(
            compare(expected[name].detach().numpy(), np.array(shared_actual[name]), atol=atol, rtol=rtol))
    return {"tokens": tokens, "context_tokens": context_tokens, "dtype": "float32",
            "mlx_device": str(mx.default_device()), "atol": atol, "rtol": rtol,
            "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="trusted local model configuration")
    parser.add_argument("--checkpoint", required=True, type=Path, help="trusted local FP32 denoiser safetensors")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=4)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args(argv)
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    report = compare_checkpoint(args.config, args.checkpoint, tokens=args.tokens,
                                context_tokens=args.context_tokens, atol=args.atol, rtol=args.rtol)
    print(json.dumps(report, allow_nan=False))
    return 0 if all(value["passed"] for value in report["comparisons"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
