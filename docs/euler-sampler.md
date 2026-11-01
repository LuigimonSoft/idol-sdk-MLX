# Preliminary Euler flow sampler

[Español](../es/euler-sampler.md)

`idol_mlx.sampler.sample_euler` implements conditional Euler
integration in the [pinned upstream sampler](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/sampler.py).
`time_grid` creates a descending sequence from 1 to 0 and applies the upstream
time shift. At each step the model receives the current sample, a float32 time
vector scaled by 1000 and the supplied condition. The next state is the current
state minus velocity times the time interval. The caller supplies initial noise;
the sampler generates no random values and leaves MLX evaluation lazy.

Optional classifier-free guidance uses `(1 + w) * v_cond - w * v_neg` when
`cfg_strength > 0` and requires `neg_condition`. The default is zero (unguided);
zero or negative values skip the negative pass, as in the upstream sampler.
`batch_cfg=True` concatenates conditional and negative inputs into one model
call per step. This is equivalent to separate calls only when the model treats
batch elements independently. The caller must provide compatible tensors and
a model. `steps` must be a positive
integer and `rescale_t` finite and positive. The equivalent denominator form
`(1-t) + rescale_t*t` avoids cancellation near the start of the grid. For very
small shifts, this intentionally preserves the mathematical endpoint even where
the upstream floating-point expression loses accuracy.

Synthetic FP32 tests compare time grids and full outputs with upstream PyTorch
using constant and time/sample-dependent toy velocities, multiple steps, shifts
and a batch of two. Guidance tests cover separate/batched modes and nonpositive
and positive strengths with a batch-independent toy model. Provisional tolerances are `atol=1e-5, rtol=1e-4` for
outputs. Tests do not validate trained models, batch-coupled model behavior, FP16/BF16,
all numeric ranges or full-model inference. Memory use of the two guidance
modes with a real model remains to be measured.

With MLX, PyTorch and the pinned source installed:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference IDOL_TEST_MLX_DEVICE=cpu \
  .venv-reference/bin/python -m unittest discover -s tests -p 'test_sampler_mlx.py' -v
```

Set `IDOL_TEST_MLX_DEVICE=gpu` for MLX Metal when available; PyTorch remains on
CPU. CI runs these parity tests on MLX CPU.
