# Preliminary Euler flow sampler

[Español](../es/euler-sampler.md)

`idol_mlx.sampler.sample_euler` implements the conditional, unguided Euler
integration in the [pinned upstream sampler](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/sampler.py).
`time_grid` creates a descending sequence from 1 to 0 and applies the upstream
time shift. At each step the model receives the current sample, a float32 time
vector scaled by 1000 and the supplied condition. The next state is the current
state minus velocity times the time interval. The caller supplies initial noise;
the sampler generates no random values and leaves MLX evaluation lazy.

This API does **not** apply classifier-free guidance or implement batched
conditional/unconditional inference; those remain separate work. The caller must
provide a compatible model, tensors and condition. `steps` must be a positive
integer and `rescale_t` finite and positive; shifts too small for the underlying
floating-point calculation are rejected.

Synthetic FP32 tests compare time grids and full outputs with upstream PyTorch
using constant and time/sample-dependent toy velocities, multiple steps, shifts
and a batch of two. Provisional tolerances are `atol=1e-5, rtol=1e-4` for
outputs. Tests do not validate trained models, guidance, FP16/BF16, all numeric
ranges or full-model inference.

With MLX, PyTorch and the pinned source installed:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_sampler_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` for MLX Metal when available; PyTorch remains on
CPU. CI runs these parity tests on MLX CPU.
