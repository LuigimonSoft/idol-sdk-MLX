# Configuration and checkpoint validation

This initial loader validates the shared structure of a local Idol model
configuration and reads local MLX `.safetensors` checkpoints against an
explicit schema. It does not download weights, convert PyTorch checkpoint
layouts or provide model-specific schemas.

## Configuration

`idol_mlx.config.load_config(path)` returns a dictionary after checking:

- `denoiser`, `decoder`, and `decoder_gs` are objects with a non-empty
  `name` and an `args` object.
- `image_size` and `max_voxel_level` are positive integers (not booleans).
- `image_encoders` contains non-empty `dinov3` and `flux2_vae` strings.

Unknown fields are preserved. The shared checks do not validate every model
argument, supported architecture, repository access or semantic compatibility;
each model must validate its own requirements when implemented. Missing,
unreadable, malformed or structurally invalid configuration raises
`ConfigError`.

## Weights

`idol_mlx.weights.load_weights(path, expected)` reads a local
`.safetensors` file using MLX. `expected` is a non-empty dictionary from
complete tensor names to `TensorSpec(shape, dtype)`. It rejects missing and
unexpected keys and any shape or dtype mismatch before returning the arrays.
Dtype names are canonical MLX names such as `float32` and `bfloat16`.
No tensors are silently cast, reshaped, mapped or transposed.

```python
from idol_mlx.weights import TensorSpec, load_weights

# Example schema only; it is not the schema of an Idol checkpoint.
expected = {"weight": TensorSpec((2, 3), "float16")}
weights = load_weights("converted-example.safetensors", expected)
```

Read or schema mismatches raise `CheckpointError`; invalid caller-provided
schemas raise `ValueError`. The caller is responsible for supplying the
complete schema derived from the corresponding model. A `.safetensors`
extension is required, and paths are trusted local inputs; this is not a
sandbox for untrusted paths. Loading does not establish provenance, license,
integrity of the original download, or numerical parity.

## Validation and remaining work

Unit tests cover configuration structure and checkpoint key, shape, dtype and
read errors using small synthetic inputs. A real MLX round-trip with synthetic
FP16/FP32 tensors and the public Idol configuration were also verified on the
reference Mac. Actual checkpoint names, trained-model schemas and conversion
rules remain to be established in issue #14. No restricted weights or
credentials are stored in this repository.

Spanish: [es/config-and-weights.md](../es/config-and-weights.md).
