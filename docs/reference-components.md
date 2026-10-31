# Reference component checks

The complete upstream pipeline has not yet been run: no Hugging Face token was
available during the access check, and unauthenticated access to the required
DINOv3 and FLUX.2 dev VAE checkpoints was denied. See
[model access and licenses](model-access.md). The issue's documented-blocker
criterion is satisfied, but end-to-end reference outputs, approved-weight
runtime shapes and numerical parity remain outstanding.

As an interim reference, `tests/test_reference_components.py` exercises small,
deterministic cases using the original Python implementation. These tests do
not load model weights, invoke the neural networks, or establish output parity
with MLX.

## Tested configuration

- Reference source: Idol SDK commit
  [4702a7f00850e34b6e83a2170cc123c3e9570744](https://github.com/speridlabs/idol-sdk/tree/4702a7f00850e34b6e83a2170cc123c3e9570744).
- Reference environment: CPython 3.13.14, PyTorch 2.15.0 and the pinned
  dependencies in `environments/reference.txt`; see
  [environment setup](environments.md).
- Device: CPU, FP32 input tensors. The checks do not require Metal, MPS, model
  weights, credentials or network access.
- Result: 3 tests passed.

## Cases and expected results

1. A toy constant-velocity model receives conditional value 2, unconditional
   value 1, initial value 10, guidance weight 3 and two Euler steps. Both
   separate and batched guidance return 5; shifted model times are 1000 and
   750. These values exercise the integration and guidance formula, not the
   real DiT.
2. Camera latent `[0, 1, 0, 0.5, 0.25]` produces yaw 0, pitch 0, radius 2
   and field of view `degrees(2 * atan(0.125))`.
3. One-hot octree child probabilities assign all seven samples to that child;
   the count remains 7 regardless of the random offset.

## Run

Set up the reference environment and pinned upstream checkout as described in
[environment setup](environments.md). From this repository's root:

```sh
IDOL_REFERENCE_SOURCE=../idol-sdk-reference .venv-reference/bin/python -m unittest discover -s tests -v
```

Set `IDOL_REFERENCE_SOURCE` to your checkout location if different. Check its
commit before testing:

```sh
git -C ../idol-sdk-reference rev-parse HEAD
```

The result must equal the commit above. If `IDOL_REFERENCE_SOURCE` is not set,
the tests are explicitly skipped; a skipped run is not validation. The tests do not check the checkout revision automatically; verify it using the command above. Missing dependencies or an unusable source may cause the tests to fail.

## Next steps

Once account access has been granted, verify access without publishing tokens,
capture the model repository revision and run the full reference pipeline.
Record exact configuration and safe, shareable artifacts separately; do not
commit credentials, private images, personal paths or model weights.

Spanish: [es/reference-components.md](../es/reference-components.md).
