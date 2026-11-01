"""Compare synthetic Euler trajectory states between PyTorch and MLX."""

import argparse
from dataclasses import asdict
import json
from importlib.metadata import version
import sys

import mlx.core as mx
import numpy as np
import torch

from idol.sampler import sample_flow
from idol_mlx.sampler import sample_euler
from scripts.parity_metrics import compare


def compare_states(reference_states, candidate_states, *, atol, rtol):
    """Return one numerical parity report per corresponding trajectory state."""
    if not reference_states or len(reference_states) != len(candidate_states):
        raise ValueError("Trajectories must have the same nonzero state count.")
    reports = []
    for step, (reference, candidate) in enumerate(zip(reference_states, candidate_states)):
        result = compare(reference, candidate, atol=atol, rtol=rtol)
        reports.append({"step": step, **asdict(result)})
    return reports


def run_synthetic(steps, *, batch_cfg=False, atol=1e-5, rtol=1e-4):
    """Compare all pre-update states and the final state of a guided toy model."""
    noise = np.array([[1.0, -0.5, 0.25], [-0.75, 0.4, 1.25]], dtype=np.float32)
    condition = np.array([[0.2, 0.5, -0.1], [-0.3, 0.4, 0.1]], dtype=np.float32)
    negative = np.array([[-0.1, 0.1, 0.3], [0.2, -0.2, 0.4]], dtype=np.float32)
    batch = noise.shape[0]
    torch_states, mlx_states = [], []
    calls_t, calls_m = [0], [0]

    def reference_model(sample, time, cond):
        if batch_cfg or calls_t[0] % 2 == 0:
            torch_states.append(sample[:batch].detach().clone())
        calls_t[0] += 1
        return 0.15 * sample + 0.001 * time[:, None] + cond

    def candidate_model(sample, time, cond):
        if batch_cfg or calls_m[0] % 2 == 0:
            mlx_states.append(sample[:batch])
        calls_m[0] += 1
        return 0.15 * sample + 0.001 * time[:, None] + cond

    reference = sample_flow(reference_model, torch.from_numpy(noise), torch.from_numpy(condition),
                            torch.from_numpy(negative), steps=steps, rescale_t=3.0,
                            cfg_strength=3.0, batch_cfg=batch_cfg, verbose=False)
    candidate = sample_euler(candidate_model, mx.array(noise), mx.array(condition), steps=steps,
                             rescale_t=3.0, neg_condition=mx.array(negative),
                             cfg_strength=3.0, batch_cfg=batch_cfg)
    torch_states.append(reference)
    mlx_states.append(candidate)
    mx.eval(*mlx_states)
    return compare_states([state.numpy() for state in torch_states],
                          [np.array(state) for state in mlx_states], atol=atol, rtol=rtol)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-cfg", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    reports = run_synthetic(args.steps, batch_cfg=args.batch_cfg, atol=args.atol, rtol=args.rtol)
    print(json.dumps({"steps": args.steps, "batch_cfg": args.batch_cfg,
                      "device": str(mx.default_device()), "mlx_version": version("mlx"),
                      "torch_version": torch.__version__, "dtype": "float32",
                      "atol": args.atol, "rtol": args.rtol, "states": reports}, allow_nan=False))
    return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
