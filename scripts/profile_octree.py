"""Measure a single trained octree anchor-sampling scenario with MLX."""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import mlx.core as mx
import numpy as np

from idol_mlx.config import load_config
from idol_mlx.octree import OctreeProbabilityFixedlenDecoder
from idol_mlx.octree_sampling import sample_octree
from idol_mlx.octree_weights import load_octree_checkpoint


def profile_checkpoint(config_path, checkpoint_path, *, num_points, level, context_tokens=32, seed=101):
    """Load trusted weights and measure one FP32 traversal, excluding setup time.

    Peak MLX memory is reset after model and input evaluation; it includes the
    still-active baseline. Process RSS is the high-water mark for the process,
    not an isolated traversal measurement. No weights or inputs are written.
    """
    if isinstance(num_points, bool) or not isinstance(num_points, int) or not 0 <= num_points < 2**24:
        raise ValueError("num_points must be an integer in [0, 2**24 - 1].")
    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 23:
        raise ValueError("level must be an integer in [0, 23].")
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int) or context_tokens < 1:
        raise ValueError("context_tokens must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer.")
    args = load_config(config_path)["decoder"]["args"]
    model = OctreeProbabilityFixedlenDecoder(**args)
    load_octree_checkpoint(model, checkpoint_path)
    rng = np.random.default_rng(seed)
    condition = mx.array(rng.standard_normal((1, context_tokens, args["cond_channels"])).astype(np.float32))
    mx.eval(model.parameters(), condition)
    mx.synchronize()
    mx.clear_cache()
    baseline_bytes = mx.get_active_memory()
    mx.reset_peak_memory()
    start = time.perf_counter()
    anchors = sample_octree(model, condition, num_points=num_points, level=level, seed=seed)
    mx.eval(anchors)
    mx.synchronize()
    elapsed = time.perf_counter() - start
    peak_bytes = mx.get_peak_memory()
    active_bytes = mx.get_active_memory()
    cache_bytes = mx.get_cache_memory()
    shape_ok = anchors.shape == (1, num_points, 3)
    finite = bool(mx.all(mx.isfinite(anchors)).item())
    in_bounds = bool(mx.all((anchors >= 0) & (anchors <= 1)).item())
    return {"num_points": num_points, "level": level, "context_tokens": context_tokens, "seed": seed,
            "dtype": "float32", "mlx_device": str(mx.default_device()), "seconds": elapsed,
            "mlx_baseline_bytes": baseline_bytes, "mlx_peak_bytes": peak_bytes,
            "mlx_active_after_bytes": active_bytes, "mlx_cache_after_bytes": cache_bytes,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "shape_ok": shape_ok, "finite": finite, "in_bounds": in_bounds}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num-points", required=True, type=int)
    parser.add_argument("--level", required=True, type=int)
    parser.add_argument("--context-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args(argv)
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    result = profile_checkpoint(args.config, args.checkpoint, num_points=args.num_points,
                                level=args.level, context_tokens=args.context_tokens, seed=args.seed)
    print(json.dumps(result, allow_nan=False))
    return 0 if all(result[key] for key in ("shape_ok", "finite", "in_bounds")) else 1


if __name__ == "__main__":
    sys.exit(main())
