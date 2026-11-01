# Comparing backend outputs

[Español](../es/parity-harness.md)

This preliminary harness compares named tensors in two trusted local
`safetensors` files using the paired metrics in
[parity metrics](parity-metrics.md). Generate the reference and candidate from
the **same captured input and weights**; independently generated random values
are not interchangeable. The helper compares outputs only: it does not run
models, load inputs into either backend, infer layer layouts, or prove that the
inputs and weights were identical. Preserve that provenance in the test itself.

Use the reference environment from [environments](environments.md), and supply
component-specific tolerances explicitly:

```sh
.venv-reference/bin/python -m scripts.compare_parity reference.safetensors candidate.safetensors \
  --reference-key component.output --candidate-key output --atol 1e-5 --rtol 1e-4
```

The command prints a JSON report with keys, tolerances, pass status, finiteness,
element count, failure count and maximum and mean absolute errors. Exit status
is 0 for a match, 1 for numerical failure (including non-finite inputs), and 2
for file, schema, shape or tolerance errors. An infinite computed error is
represented by the string `"infinity"` so the report stays valid JSON. Tolerances
are provisional until calibrated for the specific operation and hardware.

Inputs must be nonempty real numeric tensors of equal shape. NumPy safetensors
loading does not support every possible dtype (in particular BF16); export
comparison outputs as FP32 after any BF16 computation, and record that cast.
Only use trusted local files. No output file is written, and existing captures
are not modified. Never commit credentials, private inputs or restricted weights.

Unit tests cover matches, mismatches, error cases and report format. Complete
component-by-component PyTorch/MLX parity still requires MLX implementations and
shared-input integration tests; this harness alone does not establish it.
