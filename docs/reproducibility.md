# Seeded noise and reproducibility

[Español](../es/reproducibility.md)

`idol_mlx.sampler.seeded_noise` generates standard normal noise using an
explicit local MLX PRNG key. It accepts a non-empty tuple of positive dimensions,
an unsigned 64-bit integer seed and `float32`, `float16` or `bfloat16` dtype.
It does not seed or advance MLX's global random stream. Pass its output directly
to `sample_euler`, or provide an explicit preexisting MLX noise tensor instead.

```python
from idol_mlx.sampler import sample_euler, seeded_noise

noise = seeded_noise((2, 16), 42)
result = sample_euler(model, noise, condition, steps=4)
```

With the same MLX version, device, shape, dtype, model, parameters, inputs and
execution settings, repeated synthetic FP32 tests produce identical noise and
sampler outputs. This is not a promise of bitwise reproducibility across MLX
versions, hardware, devices, precision settings or models. MLX and PyTorch do
not necessarily generate the same random values for the same integer seed;
for cross-framework parity use the same explicit captured tensor instead.
Other stochastic model operations require their own explicit randomness control.

The tests cover repeated and different seeds, output dtypes, invalid inputs,
global RNG isolation and repeated sampling with a deterministic toy model.
Real models, model stochasticity and BF16/FP16 full trajectories remain untested.

With MLX, PyTorch and the pinned source installed:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_sampler_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` for MLX Metal when available; PyTorch runs on CPU.
CI runs the sampler test file on MLX CPU.
