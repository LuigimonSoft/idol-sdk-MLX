# Synthetic sampler trajectories

[Español](../es/sampler-trajectories.md)

`scripts.sampler_trajectory` compares each pre-update state and the final state
of a deterministic toy sampler run against the pinned PyTorch reference. Both
backends receive identical explicit FP32 noise and conditions. The toy velocity
depends on sample, time and condition; guidance strength and time shift are 3.
Separate and batched guidance are supported. These synthetic runs do not
validate the trained denoiser, the generated 3D result, stochastic model
operations or production performance.

Install the pinned reference environment and MLX 0.32.3 in the same environment;
set `PYTHONPATH` to the pinned upstream checkout. Run from the repository root:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.sampler_trajectory --steps 50 --batch-cfg
```

Use `--steps 2` for a short sequence or omit `--batch-cfg` for separate model
calls. `--atol` and `--rtol` set tolerances (defaults `1e-5` and `1e-4`). The
JSON report contains one entry per state, including initial and final, with
maximum and mean absolute error, failed count and pass status, as well as MLX
and PyTorch versions, MLX device, dtype and tolerances. Redirect output to a
trusted local path if a record is needed; do not commit private inputs or weights.
Exit status is 0 if every state passes and 1 if a numerical comparison fails.

The unit test exercises two- and 50-step trajectories in both modes and checks
that a deliberately changed state is reported as a failure. It runs in CI on
MLX CPU. A local Metal run of 50 steps with batched guidance on the development
machine reported 51 passing states and maximum absolute difference `0.0` for
this particular toy model; this observation is not a general accuracy guarantee.
