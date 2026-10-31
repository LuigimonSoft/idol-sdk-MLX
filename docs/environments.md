# Reproducible development environments

This repository is at the preparation stage. The MLX pipeline is not implemented yet.
These separate environments establish an initial macOS ARM64 baseline for issue #2.
They do not establish model-output parity or prove that end-to-end inference works.

## Tested configuration

- Apple Silicon Mac mini, M4, 32 GB unified memory; macOS 27.2 (26B123), arm64.
- CPython 3.13.14 and uv 0.12.1.
- The reference requirements were resolved from Idol SDK commit
  [4702a7f00850e34b6e83a2170cc123c3e9570744](https://github.com/speridlabs/idol-sdk/tree/4702a7f00850e34b6e83a2170cc123c3e9570744).
- MLX 0.32.3 and mlx-metal 0.32.3. PyTorch 2.15.0, torchvision 0.29.2,
  Transformers 5.22.1, and Diffusers 0.42.1. All transitive packages and
  artifact hashes are in the requirements files.

These locks are intended for the tested Python and platform. Other configurations
require a separate resolution and validation. A different Python patch release is
not asserted to be reproducible.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and make
CPython 3.13.14 available, for example with `uv python install 3.13.14`.
From this repository's root:

```sh
uv venv --python 3.13.14 .venv-reference
uv pip sync --python .venv-reference/bin/python --require-hashes environments/reference.txt

uv venv --python 3.13.14 .venv-mlx
uv pip sync --python .venv-mlx/bin/python --require-hashes environments/mlx.txt
```

The reference file installs the dependencies of the cited upstream commit, not
the upstream source itself. To compare against the original SDK, obtain that
commit in a separate checkout and use its source path, for example:

```sh
git clone https://github.com/speridlabs/idol-sdk.git ../idol-sdk-reference
git -C ../idol-sdk-reference checkout 4702a7f00850e34b6e83a2170cc123c3e9570744
PYTHONPATH=../idol-sdk-reference .venv-reference/bin/python -c "import idol; from diffusers.models import AutoencoderKLFlux2; from transformers import AutoModel; from torchvision.ops import deform_conv2d; print('Reference imports OK')"
```

For MLX and Metal:

```sh
.venv-mlx/bin/python -c "import mlx.core as mx; assert mx.metal.is_available(); assert mx.default_device() == mx.gpu; x = mx.array([1., 2., 3.]) * 2; mx.eval(x); assert x.tolist() == [2., 4., 6.]; print('Metal OK')"
```

Metal may be unavailable inside a sandbox or headless session even when the
machine supports it; run the check in a normal local terminal. PyTorch MPS
availability can be checked with
`.venv-reference/bin/python -c "import torch; print(torch.backends.mps.is_available())"`.
The original pipeline selects CUDA or CPU automatically and does not select MPS
automatically; this environment setup does not change that behavior.

## Validation and limitations

The two locked environments were installed with hash verification. The reference
imports above passed; MLX executed the shown calculation on Metal, and PyTorch
reported MPS available outside the sandbox. No gated model weights were
downloaded and no full model inference or output-parity test has been performed.
Access to the DINOv3 and FLUX.2 model repositories and their terms remains a
separate task.

Spanish translation: [es/environments.md](../es/environments.md).
