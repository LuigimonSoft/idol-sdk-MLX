# Synthetic tensor reference capture

This is a partial reference for migration issue #5. It captures **synthetic**
inputs and randomly initialized weights from selected upstream components; it
does not include trained checkpoints, model input images, credentials or
personal data. It does not establish MLX parity or complete the full-pipeline
reference task.

The generator uses the original SDK at
[commit 4702a7f00850e34b6e83a2170cc123c3e9570744](https://github.com/speridlabs/idol-sdk/tree/4702a7f00850e34b6e83a2170cc123c3e9570744)
and the CPU reference environment in [environments.md](environments.md).
Outputs may differ across PyTorch versions or platforms; compare backends using
the **same captured file**, rather than assuming shared random seeds produce
identical inputs or weights.

## Included tensors

| Prefix | Captured data | Scope |
| --- | --- | --- |
| `repo.*` | Explicit FP32 input, real-valued rotary output and every state-dict tensor of a small randomly initialized RePo module. | Width 24, 2 heads, 3 tokens. |
| `attention.*` | Explicit input, rotary input, attention output and all state-dict tensors of a small randomly initialized RoPE attention module. | Width 24, 2 heads, 3 tokens. |
| `sampler.*` | Explicit initial noise, conditional and negative values and final output. | Two Euler steps, shift 3, guidance 3; toy constant-velocity model. |
| `camera.*` | Five input values and output in yaw, pitch, radius, field-of-view order. | A known perspective-camera example; output stored as FP64. |

Each weight key uses `<module>.weights.<state_dict_key>`; input and output keys
are explicit. Only the listed components are covered. VAE randomness, trained
weights, the DiT, both full decoders, image encoders and the full pipeline remain
to be captured and validated in later work.

## Generate and test

Set up the upstream checkout and pinned reference environment using
[environments.md](environments.md). Confirm the checkout revision matches the
commit above. From this repository root, set the source path to your checkout:

```sh
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python scripts/capture_reference.py --output /tmp/idol-synthetic-reference.safetensors
IDOL_REFERENCE_SOURCE=../idol-sdk-reference .venv-reference/bin/python -m unittest discover -s tests -p 'test_capture_reference.py' -v
```

Use a new output path: the script refuses to overwrite a file. It saves a
standard safetensors file that can be loaded by either backend; all model
weights in this file are synthetic. The tests check tensor shapes, expected toy
sampler and camera values, safe serialization, overwrite protection and
reconstruction of component outputs from captured weights. If
`IDOL_REFERENCE_SOURCE` is absent, the tests are skipped; that is not
validation. The tests do not automatically verify the source revision.

Validated on CPU using CPython 3.13.14, PyTorch 2.15.0 and safetensors 0.8.0:
both tests passed. Actual tensor outputs are generated locally and are not
committed. Capture artifacts should be reviewed before sharing; never commit
credentials, private images, personal paths or restricted model weights.

Spanish: [es/reference-capture.md](../es/reference-capture.md).
