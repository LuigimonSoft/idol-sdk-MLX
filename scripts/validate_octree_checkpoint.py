"""Compare a reduced FP32 octree forward pass with a trusted checkpoint."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
import torch
from safetensors.torch import load_file

from idol.models.decoder import OctreeProbabilityFixedlenDecoder as TorchDecoder
from idol.models.layers import PcdAbsolutePositionEmbedderv2 as TorchPositionEmbedder
from idol_mlx.config import load_config
from idol_mlx.octree import OctreeProbabilityFixedlenDecoder
from idol_mlx.octree_weights import load_octree_checkpoint
from scripts.parity_metrics import compare


def compare_checkpoint(config_path, checkpoint_path, *, cells=8, context_tokens=4, seed=2026,
                       level=4, atol=2e-5, rtol=2e-4):
    """Compare shared synthetic inputs; no inputs or weights are written."""
    for name, value in (("cells", cells), ("context_tokens", context_tokens), ("level", level)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer.")
    args = load_config(config_path)["decoder"]["args"]
    # Non-persistent position frequencies must be rebuilt after meta construction.
    with torch.device("meta"):
        reference = TorchDecoder(**args)
    reference.load_state_dict(load_file(str(checkpoint_path)), strict=True, assign=True)
    reference.pos_embedder = TorchPositionEmbedder(channels=args["model_channels"], in_channels=3)
    reference.dtype = torch.float32
    reference.eval()
    candidate = OctreeProbabilityFixedlenDecoder(**args)
    load_octree_checkpoint(candidate, checkpoint_path)
    rng = np.random.default_rng(seed)
    coords = rng.random((1, cells, 3), dtype=np.float32)
    levels = np.array([level], dtype=np.float32)
    cond = rng.standard_normal((1, context_tokens, args["cond_channels"])).astype(np.float32)
    with torch.no_grad():
        expected_position = reference.pos_embedder(torch.from_numpy(coords).reshape(-1, 3)).numpy()
        expected = reference(torch.from_numpy(coords), torch.from_numpy(levels), torch.from_numpy(cond))
        expected_probs = torch.softmax(expected, dim=-1).numpy()
    actual_position = candidate.pos_embedder(mx.array(coords).reshape(-1, 3))
    actual = candidate(mx.array(coords), mx.array(levels), mx.array(cond))
    actual_probs = mx.softmax(actual, axis=-1)
    mx.eval(actual_position, actual, actual_probs)
    comparisons = {
        "position": asdict(compare(expected_position, np.array(actual_position), atol=atol, rtol=rtol)),
        "logits": asdict(compare(expected.numpy(), np.array(actual), atol=atol, rtol=rtol)),
        "probabilities": asdict(compare(expected_probs, np.array(actual_probs), atol=atol, rtol=rtol)),
    }
    return {"cells": cells, "context_tokens": context_tokens, "seed": seed, "level": level,
            "dtype": "float32", "mlx_device": str(mx.default_device()), "atol": atol, "rtol": rtol,
            "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args(argv)
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    report = compare_checkpoint(args.config, args.checkpoint, cells=args.cells,
                                context_tokens=args.context_tokens, seed=args.seed, level=args.level,
                                atol=args.atol, rtol=args.rtol)
    print(json.dumps(report, allow_nan=False))
    return 0 if all(item["passed"] for item in report["comparisons"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
