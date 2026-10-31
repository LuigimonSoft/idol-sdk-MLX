# MLX backend architecture proposal

This document defines the initial design for issue #9. It is a proposed
interface, not an implemented or validated API. The reference behavior and
model dimensions are documented in [architecture.md](architecture.md).
The original [Idol SDK](https://github.com/speridlabs/idol-sdk) remains a
separate PyTorch reference checkout for parity tests.

## Boundaries

- Implement the new backend as a separate `idol_mlx` package. Do not import
  PyTorch, torchvision, Diffusers, Transformers or TensorDict in the MLX
  runtime. Those packages may remain in the separate reference environment or
  an explicitly separate conversion tool when needed.
- Use MLX arrays inside model and sampling stages. Pillow image handling and a
  backend-independent PLY serialization boundary may use ordinary Python and
  NumPy. Avoid implicit per-step conversion between frameworks.
- Preserve the existing user-facing concepts: image input, synthetic/real crop
  modes, seed, steps, classifier-free guidance, time shift, Gaussian count,
  preprocessed image, camera and PLY export. Compatibility is a goal to
  validate, not a claim of drop-in behavior.
- The first supported end-to-end path targets a single image. Internal batch
  dimensions are retained for future work but batching is not promised.

## Proposed package organization

These paths describe responsibilities; create files only when their respective
issues are implemented.

| Module | Responsibility | Direct dependencies / boundary |
| --- | --- | --- |
| `idol_mlx.pipeline` | Own configuration, component lifecycle and orchestration; expose load, preprocess and generate operations. | MLX components, input/output types, configuration; no PyTorch. |
| `idol_mlx.config` | Parse and validate model and generation configuration, dimensions and supported values. | Python and typed data; no device work or secret storage. |
| `idol_mlx.weights` | Locate a local converted checkpoint; validate manifest, names, shapes and dtypes before loading. | MLX, local files; optional Hub access isolated from model definitions. |
| `idol_mlx.preprocess` | Image conversion, alpha detection, crop, resize and composite. | Pillow/NumPy; optional segmentation interface. |
| `idol_mlx.models.birefnet` | Produce a segmentation mask for inputs without useful alpha. | MLX layers and compatible weights; deformable convolution remains an open design risk. |
| `idol_mlx.conditioning` and encoder modules | Produce DINOv3 and FLUX.2 VAE features, including required VAE packing and normalization. | MLX, model-specific checkpoints; their external access and licenses remain separate. |
| `idol_mlx.models.layers` | Linear layers, normalization, embeddings, attention and rotary operations. | MLX only; per-component parity tests. |
| `idol_mlx.models.dit` | Predict latent and camera velocities for a time and image condition. | MLX shared layers and denoiser weights. |
| `idol_mlx.sampler` | Euler time schedule and classifier-free guidance around a velocity model. | MLX arrays, a velocity callable and explicit random inputs. |
| `idol_mlx.models.decoder` | Octree probabilities, anchor sampling and Gaussian attributes. | MLX shared layers, decoder weights and controlled random inputs. |
| `idol_mlx.camera` and `idol_mlx.gaussian` | Convert outputs to camera values and a Gaussian representation; export standard PLY. | MLX at numeric boundary, host serialization only at export. |
| `idol_mlx.sample` | CLI argument parsing and output file selection. | Public pipeline API; paths selected by a trusted local user. |

Conversion of upstream checkpoints is a separate concern: preserve source
provenance and make conversion explicit. Loading must fail clearly on missing,
unexpected or incompatible tensors instead of silently accepting partial
weights. Exact mappings and convolution layouts belong to issue #14.

## Component contracts

All shapes below describe the known reference configuration, not a promise for
other checkpoints. `B` is batch size, `L` is the common image-token length,
`A` is the number of anchors and `G=32A`. Dtypes and actual feature lengths
must be verified against captured trained-model results.

| Boundary | Input | Output |
| --- | --- | --- |
| Preprocess | Image or local image path, mode and optional margin; segmentation component when needed. | Prepared RGB image and pixel array `[B, 3, 1024, 1024]` in [0, 1]. |
| Image conditioning | Pixel array plus controlled VAE random draw/key. | Named features `feature1 [B, L, 1280]` and `feature2 [B, L, 128]`; token order and preprocessing match the reference. |
| DiT | Named object latent `[B, 8192, 16]`, camera latent `[B, 1, 5]`, time `[B]` and image features. | Named velocity arrays with the same latent and camera shapes. |
| Sampler | Velocity function, named initial noise, positive and negative features, schedule and guidance parameters. | Sampled object and camera latents; optional debug capture must not silently expose user inputs. |
| Octree | Object latent, anchor count, level and controlled random inputs. | Anchor positions `[B, A, 3]`; exact counts and ordering verified separately. |
| Gaussian decoder | Anchors and object latent. | Per-anchor raw values `[B, A, 480]` and transformed attributes for `G` Gaussians. |
| Camera / export | Camera latent `[5]` and Gaussian attributes. | Camera fields (yaw, pitch, radius, field of view) and PLY with the reference conventions. |

The reference pipeline computes `A=max(1, num_gaussians // 32)`. The public
contract should document supported limits and rounding only after behavior has
been tested; do not silently assert that arbitrary requested counts are honored.

## Public API sketch

Names and signatures are tentative until integration tests define compatibility:

```python
# Proposed only: this package and API do not exist yet.
from idol_mlx import IdolPipeline

pipeline = IdolPipeline.from_pretrained("speridlabs/idol")
result = pipeline("object.png", mode="real", seed=42, steps=50)
result.gaussian.save_ply("object.ply")
result.preprocessed_image.save("object_input.png")
# result.camera contains the predicted camera fields.
```

The load operation must distinguish missing files, missing gated access,
unsupported device, invalid config and incompatible checkpoint. It must not
silently fall back from MLX to CPU or PyTorch while claiming Metal execution.
No token is committed, logged or embedded in configuration; use the Hub's
standard authentication or an environment secret.

## Evaluation, precision and randomness

MLX evaluates lazily. Evaluate tensors at deliberate boundaries for benchmark
timing, serialization, memory measurement and comparisons; do not insert
synchronization on every operation without measuring the cost. See
[MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
and [unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html).

Carry dtype decisions explicitly per component and preserve required FP32
normalization boundaries. FP16/BF16 support, memory peaks and parity must be
measured on the target device. Treat compilation as a later optimization:
dynamic octree shapes, RNG state, memory and recompilation behavior require
separate tests.

Own randomness per generation rather than relying on hidden cross-framework
seed equivalence. Use explicit MLX PRNG keys or injectable draws for the VAE,
initial latent and camera noise, octree sampling and leaf jitter. A parity
fixture supplies the same tensors/draws to both backends. Record algorithm and
version as needed; a seed alone is not a cross-framework reference. See
[MLX random keys](https://ml-explore.github.io/mlx/build/html/python/random.html).

## Integration and validation order

1. Implement packaging and Metal detection (issues #10 and #12).
2. Implement strict config/weight loading and conversion mapping (#13–#14).
3. Port shared layers and verify against synthetic captured inputs and weights
   (#17–#21); use the provisional [parity policy](parity-metrics.md).
4. Port DiT, sampler, camera and decoders with explicit noise (#22–#36).
5. Integrate image encoders and alpha-first preprocessing before BiRefNet
   (#37–#49). Review component licenses and approved access first.
6. Validate the complete pipeline, PLY and CLI, then measure speed and memory
   before optimizing or deciding whether to compile (#50–#64).

Each boundary requires shape, dtype, error and determinism tests; full inference
and quality comparisons require gated weights and are not part of this design.
A future service or agent accepting untrusted paths must implement a separate
security boundary; see [output path security](output-path-security.md).

Spanish: [es/mlx-architecture.md](../es/mlx-architecture.md).
