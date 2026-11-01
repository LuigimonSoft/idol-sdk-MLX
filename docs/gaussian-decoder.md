# Preliminary FP32 Gaussian feature decoder

[Español](../es/gaussian-decoder.md)

`idol_mlx.gaussian_decoder.GaussianFixedlenDecoder` implements the FP32 feature
forward path of the [pinned upstream Gaussian decoder](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/idol/models/decoder.py).
It takes anchor coordinates `[B, L, 3]` and conditioning `[B, Lc, C]`, and
returns raw features `[B, L, num_gaussians * 15]`. Conversion of these features
into position, scale, rotation, opacity and color is provided separately by
[Gaussian attribute conversion](gaussian-attributes.md); raw features alone
are not a renderable 3D representation.

`idol_mlx.gaussian_weights.load_gaussian_checkpoint` strictly validates the
complete upstream-named FP32 safetensors schema before changing the model,
including the scalar `base_offset_scale` and `points_offset_perturbation`
buffers. No weights are cast or ignored. The API does not download weights.
The model supports 3D positions and explicitly FP32 inference; upstream mixed
precision is not validated. Controlled synthetic weights and inputs pass
reference comparisons on CPU and Metal.

## Reduced trained-checkpoint comparison

The strictly loaded `decoder_gs.safetensors` (328 FP32 tensors) from
[`speridlabs/idol` revision `2731c2291fc65a797c5cb627a5e5526fc1490902`](https://huggingface.co/speridlabs/idol/tree/2731c2291fc65a797c5cb627a5e5526fc1490902)
was compared against PyTorch explicitly run in FP32 on an Apple M4 with 32 GiB,
macOS 27.2, Python 3.13.14, MLX 0.32.3 and PyTorch 2.15.0. Each run used batch 1
and synthetic FP32 inputs. The provisional tolerance was `atol=2e-5`,
`rtol=2e-4`; relative error affects whether a value passes.

| MLX device | Anchors | Context tokens | Seed | Feature max absolute error | Failed features | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU | 8 | 4 | 2026 | 1.33514e-5 | 0 / 3840 | Pass |
| Metal GPU | 8 | 4 | 2026 | 2.47955e-5 | 0 / 3840 | Pass |
| CPU | 32 | 8 | 42 | 1.64509e-5 | 0 / 15360 | Pass |
| Metal GPU | 32 | 8 | 42 | 5.25117e-5 | 6 / 15360 | **Fail** |

Position embeddings passed in all four cases. Setting `MLX_ENABLE_TF32=0` did
not change the observed GPU failure for the 32-anchor input. The cause of the
remaining feature difference is not established; issue #35 remains open and
tolerances have not been increased. These reduced inputs do not validate full
anchor counts, attribute conversion, mixed precision or end-to-end inference.
No weights or inputs were committed.

## Attention and block diagnostics

An optional reduced diagnostic compares intermediate block outputs and runs an
additional forward pass with explicit FP32 matrix multiplications and
`mx.softmax(..., precise=True)` instead of fused MLX attention. For the
32-anchor / 8-context / seed-42 input, the first block whose elements exceed
the same provisional tolerance is block **3** on Metal and block **6** on
CPU (zero-based). These block checks are diagnostic only; they are not separate
acceptance criteria, and failures can disappear after later layers. The first
failing block can include errors accumulated earlier; it is not proof that the
block itself caused the discrepancy.

| Metal scenario (anchors / context / seed) | Fused failed values | Explicit failed values | Fused max error | Explicit max error |
| --- | ---: | ---: | ---: | ---: |
| 32 / 8 / 42 | 6 / 15360 | 0 / 15360 | 5.25117e-5 | 3.05176e-5 |
| 64 / 16 / 123 | 35 / 30720 | 26 / 30720 | 3.70026e-4 | 3.49045e-4 |

The explicit calculation improves these cases but does not resolve the second
one, and does not establish a general fix or the sole cause. It materializes
attention scores and can require quadratic memory. The diagnostic is limited
to at most 256 anchors and 256 condition tokens; it temporarily patches the
attention function process-wide and must not run concurrently with other MLX
work in the same process. Production inference continues to use fused attention.
The command's exit status remains based only on the original position and
fused-feature comparisons, not optional diagnostic metrics.

## Reproduce

With the pinned upstream SDK on `PYTHONPATH`, use trusted local configuration
and weights obtained under their applicable terms. The diagnostic allocates the
full reference and MLX models and prints only aggregate metrics. Exit 0 means
both comparisons passed; exit 1 means at least one failed.

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -m scripts.validate_gaussian_checkpoint \
  --config /trusted/path/config.json --checkpoint /trusted/path/decoder_gs.safetensors \
  --anchors 32 --context-tokens 8 --seed 42 --device gpu
```

Use `--device cpu` for MLX CPU. Add `--trace-blocks --compare-attention` for
the optional diagnostic metrics; these flags do not change the production model
or command exit criteria. The trained checkpoint is not needed by CI.
