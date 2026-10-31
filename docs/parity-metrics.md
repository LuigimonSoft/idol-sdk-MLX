# Numerical parity metrics

This document establishes an initial comparison method for issue #7. The
thresholds below are **provisional starting points**, not measured guarantees
or final acceptance criteria. Calibrate each component against captured
references on the target hardware and record the observed errors before adopting
a threshold. An out-of-tolerance result signals investigation; it does not by
itself identify the cause.

## Deterministic tensor comparisons

Given a PyTorch reference `r` and MLX candidate `c`, compare identical
shapes and inputs, using the captured values rather than independently generated
random numbers. For each element, require:

```text
abs(c - r) <= atol + rtol * abs(r)
```

Use absolute tolerance near zero and relative tolerance for larger values.
The helper in `scripts/parity_metrics.py` also reports maximum and mean
absolute error, the total number of elements and the number outside tolerance.
It casts real numeric arrays to FP64 for the calculation. This cannot recover
precision already lost in the input tensors. Integer values outside the guaranteed
exact FP64 range [-2**53, 2**53] are rejected before conversion. Mean error is
computed with scaling to avoid overflow when individual errors are finite.

Shape mismatch, empty arrays, invalid tolerances and non-real or non-numeric
inputs are rejected. Any NaN or infinity fails even if both inputs contain the
same value; in that case the helper reports the non-finite element count and
does not compute aggregate error. Always validate shapes and finiteness
separately from numeric closeness.

## Stage-specific starting policy

| Stage | Initial comparison | Tolerance / acceptance status |
| --- | --- | --- |
| Layout, shapes, keys, token order, camera JSON keys and counts | Exact structural checks. | Exact; no numerical tolerance. |
| FP32 embeddings, normalization, rotary operations, camera and toy sampler | Paired element-wise comparison plus max/mean error. | Exploratory `atol=1e-5`, `rtol=1e-4`; calibrate per operation. Camera fields have different units and should be assessed individually. |
| FP16/BF16 attention, transformer blocks and decoders | Paired intermediate outputs at the same precision and input; record error by stage. | No approved threshold yet. Measure the target hardware and choose component-specific tolerances. |
| Euler trajectory | Compare individual velocities and intermediate states using identical initial noise, conditions and time schedule. | No final threshold; report error growth per step rather than assuming the FP32 per-operation values apply to the full trajectory. |
| VAE posterior and octree sampling | First compare deterministic computations with explicitly shared stochastic draws, where the implementations support injecting them. | No pointwise pass criterion until randomness is controlled; do not compare unrelated same-seed streams. |
| Octree counts and Gaussian output | Check count conservation, finiteness and valid attributes; with shared randomness compare corresponding values only when ordering is controlled. | Exact invariants plus calibrated numeric checks; ordering or stochastic variation may require separate distributional tests. |
| Full image-to-3D output | Compare outputs and rendered views using fixed, shareable inputs and explicitly documented settings. | Visual and distributional acceptance metrics remain to be defined and validated; a single PLY element-wise threshold is insufficient. |

The upstream RePo and attention synthetic capture can provide an early shared
input and weights; it is not a trained-model result. Avoid reporting a
comparison as validated if reference data or MLX execution is missing.

## Verification

With the pinned reference environment installed, run from the repository root:

```sh
.venv-reference/bin/python -m unittest discover -s tests -p 'test_parity_metrics.py' -v
```

The tests cover exact equality, near-zero absolute tolerance, relative
tolerance, summary metrics, non-finite values, shapes and invalid inputs. The
helper evaluates arrays supplied by a caller; converting tensors from PyTorch or
MLX, defining policies for each component and capturing real-model references
remain separate work. Do not commit credentials, private inputs or model weights.

Spanish: [es/parity-metrics.md](../es/parity-metrics.md).
