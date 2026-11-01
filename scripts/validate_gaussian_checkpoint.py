"""Compare a reduced FP32 Gaussian feature forward pass with a trusted checkpoint."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
import torch
from safetensors.torch import load_file

from idol.models.decoder import GaussianFixedlenDecoder as TorchDecoder
from idol_mlx.config import load_config
from idol_mlx.gaussian_decoder import GaussianFixedlenDecoder
from idol_mlx.gaussian_weights import load_gaussian_checkpoint
from scripts.parity_metrics import compare


def compare_checkpoint(config_path, checkpoint_path, *, anchors=8, context_tokens=4, seed=2026,
                       atol=2e-5, rtol=2e-4):
    """Compare raw feature tensors and position embeddings with shared synthetic inputs.

    Both full models are allocated; the reference is run explicitly in FP32.
    No checkpoint tensors, inputs or predicted features are written.
    """
    for name, value in (("anchors", anchors), ("context_tokens", context_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer.")
    args = load_config(config_path)["decoder_gs"]["args"]
    reference = TorchDecoder(**args)
    reference.load_state_dict(load_file(str(checkpoint_path)), strict=True, assign=True)
    reference.dtype = torch.float32
    reference.eval()
    candidate = GaussianFixedlenDecoder(**args)
    load_gaussian_checkpoint(candidate, checkpoint_path)
    rng = np.random.default_rng(seed)
    points = rng.random((1, anchors, 3), dtype=np.float32)
    condition = rng.standard_normal((1, context_tokens, args["cond_channels"])).astype(np.float32)
    with torch.no_grad():
        expected_pos = reference.pos_embedder(torch.from_numpy(points).reshape(-1, 3)).numpy()
        expected = reference(torch.from_numpy(points), torch.from_numpy(condition)).numpy()
    actual_pos = candidate.pos_embedder(mx.array(points).reshape(-1, 3))
    actual = candidate(mx.array(points), mx.array(condition))
    mx.eval(actual_pos, actual)
    comparisons = {
        "position": asdict(compare(expected_pos, np.array(actual_pos), atol=atol, rtol=rtol)),
        "features": asdict(compare(expected, np.array(actual), atol=atol, rtol=rtol)),
    }
    return {"anchors": anchors, "context_tokens": context_tokens, "seed": seed,
            "dtype": "float32", "mlx_device": str(mx.default_device()), "atol": atol, "rtol": rtol,
            "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args(argv)
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    report = compare_checkpoint(args.config, args.checkpoint, anchors=args.anchors,
                                context_tokens=args.context_tokens, seed=args.seed,
                                atol=args.atol, rtol=args.rtol)
    print(json.dumps(report, allow_nan=False))
    return 0 if all(result["passed"] for result in report["comparisons"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
