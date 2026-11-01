"""Compare a reduced FP32 Gaussian feature forward pass with a trusted checkpoint."""

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

from idol.models.decoder import GaussianFixedlenDecoder as TorchDecoder
from idol_mlx.config import load_config
from idol_mlx.gaussian_decoder import GaussianFixedlenDecoder
from idol_mlx.gaussian_weights import load_gaussian_checkpoint
from scripts.parity_metrics import compare


def compare_checkpoint(config_path, checkpoint_path, *, anchors=8, context_tokens=4, seed=2026,
                       atol=2e-5, rtol=2e-4, trace_blocks=False, compare_attention=False):
    """Compare raw feature tensors and position embeddings with shared synthetic inputs.

    Both full models are allocated; the reference is run explicitly in FP32.
    No checkpoint tensors, inputs or predicted features are written.
    """
    for name, value in (("anchors", anchors), ("context_tokens", context_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer.")
    if not isinstance(trace_blocks, bool) or not isinstance(compare_attention, bool):
        raise ValueError("diagnostic flags must be boolean.")
    if (trace_blocks or compare_attention) and (anchors > 256 or context_tokens > 256):
        raise ValueError("diagnostic modes require anchors and context_tokens at most 256.")
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
    reference_blocks = []
    hooks = []
    if trace_blocks:
        def capture_block(_module, _inputs, output):
            reference_blocks.append(output.detach().numpy().copy())
        hooks = [block.register_forward_hook(capture_block) for block in reference.blocks]
    try:
        with torch.no_grad():
            expected_pos = reference.pos_embedder(torch.from_numpy(points).reshape(-1, 3)).numpy()
            expected = reference(torch.from_numpy(points), torch.from_numpy(condition)).numpy()
    finally:
        for hook in hooks:
            hook.remove()
    points_m, condition_m = mx.array(points), mx.array(condition)
    actual_pos = candidate.pos_embedder(points_m.reshape(-1, 3))
    actual = candidate(points_m, condition_m)
    mx.eval(actual_pos, actual)
    comparisons = {
        "position": asdict(compare(expected_pos, np.array(actual_pos), atol=atol, rtol=rtol)),
        "features": asdict(compare(expected, np.array(actual), atol=atol, rtol=rtol)),
    }
    report = {"anchors": anchors, "context_tokens": context_tokens, "seed": seed,
              "dtype": "float32", "mlx_device": str(mx.default_device()), "atol": atol, "rtol": rtol,
              "comparisons": comparisons}
    if trace_blocks:
        if len(reference_blocks) != len(candidate.blocks):
            raise RuntimeError("reference and MLX block counts differ.")
        hidden = candidate.input_layer(candidate.in_proj(points_m) + actual_pos.reshape(1, anchors, -1))
        block_results = []
        for index, (block, expected_block) in enumerate(zip(candidate.blocks, reference_blocks)):
            hidden = block(hidden, condition_m)
            mx.eval(hidden)
            block_results.append({"index": index, **asdict(compare(expected_block, np.array(hidden),
                                                                    atol=atol, rtol=rtol))})
        report["block_diagnostics"] = block_results
    if compare_attention:
        def explicit_attention(q, k, v, *, scale):
            scores = (q * scale) @ k.transpose(0, 1, 3, 2)
            return mx.softmax(scores, axis=-1, precise=True) @ v
        # Diagnostic only: this process-wide patch is unsuitable for concurrent use.
        with patch.object(mx.fast, "scaled_dot_product_attention", side_effect=explicit_attention):
            alternate = candidate(points_m, condition_m)
            mx.eval(alternate)
        report["explicit_attention_diagnostic"] = asdict(compare(expected, np.array(alternate),
                                                                   atol=atol, rtol=rtol))
    return report


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
    parser.add_argument("--trace-blocks", action="store_true", help="report per-block diagnostic metrics")
    parser.add_argument("--compare-attention", action="store_true",
                        help="compare explicit attention for reduced inputs (diagnostic only)")
    args = parser.parse_args(argv)
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    report = compare_checkpoint(args.config, args.checkpoint, anchors=args.anchors,
                                context_tokens=args.context_tokens, seed=args.seed,
                                atol=args.atol, rtol=args.rtol, trace_blocks=args.trace_blocks,
                                compare_attention=args.compare_attention)
    print(json.dumps(report, allow_nan=False))
    return 0 if all(result["passed"] for result in report["comparisons"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
