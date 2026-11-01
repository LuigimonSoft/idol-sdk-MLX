"""Compare reduced octree traversal using controlled offsets and leaf jitter."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import torch
from safetensors.torch import load_file

from idol.models.decoder import OctreeProbabilityFixedlenDecoder as TorchDecoder
from idol.models.layers import PcdAbsolutePositionEmbedderv2 as TorchPositionEmbedder
from idol_mlx.config import load_config
from idol_mlx.octree import OctreeProbabilityFixedlenDecoder
from idol_mlx.octree_sampling import sample_octree
from idol_mlx.octree_weights import load_octree_checkpoint
from scripts.parity_metrics import compare


def compare_checkpoint(config_path, checkpoint_path, *, num_points=32, level=2, context_tokens=4, seed=101):
    """Compare anchor coordinates with all offsets and jitters set to 0.375.

    The reference and MLX RNG APIs are patched only within this diagnostic; no
    trained weights or generated inputs are written. FP32 inference is used.
    """
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int) or context_tokens < 1:
        raise ValueError("context_tokens must be a positive integer.")
    if isinstance(num_points, bool) or not isinstance(num_points, int) or not 1 <= num_points < 2**24:
        raise ValueError("num_points must be an integer in [1, 2**24 - 1].")
    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 23:
        raise ValueError("level must be an integer in [0, 23].")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer.")
    args = load_config(config_path)["decoder"]["args"]
    with torch.device("meta"):
        reference = TorchDecoder(**args)
    reference.load_state_dict(load_file(str(checkpoint_path)), strict=True, assign=True)
    reference.pos_embedder = TorchPositionEmbedder(channels=args["model_channels"], in_channels=3)
    reference.dtype = torch.float32
    reference.eval()
    candidate = OctreeProbabilityFixedlenDecoder(**args)
    load_octree_checkpoint(candidate, checkpoint_path)
    rng = np.random.default_rng(seed)
    cond = rng.standard_normal((1, context_tokens, args["cond_channels"])).astype(np.float32)

    def fixed_torch_rand(*shape, device=None):
        return torch.full(shape, 0.375, dtype=torch.float32, device=device)

    def fixed_torch_jitter(value, dtype=None):
        return torch.full_like(value, 0.375, dtype=dtype)

    def fixed_mlx_uniform(low=0, high=1, shape=(), dtype=None, key=None, stream=None):
        return mx.full(shape, 0.375, dtype=dtype)

    with patch("torch.rand", side_effect=fixed_torch_rand), patch("torch.rand_like", side_effect=fixed_torch_jitter):
        expected = TorchDecoder.sample(reference, torch.from_numpy(cond), num_points, level).numpy()
    with patch.object(mx.random, "uniform", side_effect=fixed_mlx_uniform):
        actual = sample_octree(candidate, mx.array(cond), num_points=num_points, level=level, seed=seed)
        mx.eval(actual)
    metrics = asdict(compare(expected, np.array(actual), atol=0, rtol=0))
    return {"num_points": num_points, "level": level, "context_tokens": context_tokens,
            "seed": seed, "offset_and_jitter": 0.375, "dtype": "float32",
            "mlx_device": str(mx.default_device()), "coordinates": metrics}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num-points", type=int, default=32)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args(argv)
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    report = compare_checkpoint(args.config, args.checkpoint, num_points=args.num_points,
                                level=args.level, context_tokens=args.context_tokens, seed=args.seed)
    print(json.dumps(report, allow_nan=False))
    return 0 if report["coordinates"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
