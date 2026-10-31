# Reproducible test images

This small public fixture set supports preprocessing and future end-to-end
functional and visual comparisons. It contains no personal images or model
weights.

| File | Type | Purpose | Source and rights |
| --- | --- | --- | --- |
| `coffee.png` | RGB photograph, 600 × 400 | Real photograph with a prominent cup and background. | [scikit-image v0.25.2](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_fetchers.py), photographer Rachel Michetti; upstream identifies it as CC0. |
| `clock_motion.png` | Grayscale photograph, 400 × 300 | Real, motion-blurred object; checks grayscale handling. | [scikit-image v0.25.2](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_fetchers.py), photographer Stefan van der Walt; upstream says it was released into the public domain. |
| `synthetic_opaque.png` | RGB, 256 × 256 | Simple opaque synthetic object; requires background removal. | Generated locally by the fixture script. |
| `synthetic_alpha.png` | RGBA, 256 × 256 | Synthetic object with transparent and opaque pixels; exercises the existing-alpha path. | Generated locally by the fixture script. |

The photographs have complex backgrounds; they are useful inputs, not
segmentation ground truth or guaranteed examples of reconstruction quality.
The reference pipeline is designed for images of a single object; the coffee
scene can expose ambiguity. Visual outputs must be reviewed independently.

## Prepare and test

Set up the pinned reference environment described in
[environments.md](environments.md). Choose a new empty destination **outside
the repository** and run:

```sh
.venv-reference/bin/python scripts/prepare_test_images.py --output /tmp/idol-test-images
.venv-reference/bin/python -m unittest discover -s tests -p 'test_prepare_test_images.py' -v
```

The generator downloads two public photographs from the specified version tag
and checks their SHA-256 digests before writing anything. It draws the synthetic
images with the pinned Pillow version. It refuses to overwrite existing fixture
names; use a fresh destination for another run. Downloading requires network
access, but the unit tests use mocked downloads and work offline. A failed
network request or digest mismatch should be investigated rather than bypassed.

The tests check image modes and alpha, rejection of an incorrect digest, creation
of all four files and refusal to overwrite. Verified on the reference
environment: all three tests passed, and an actual download produced the
sizes and modes above. A deterministic image fixture does not guarantee
deterministic model inference.

Spanish: [es/test-images.md](../es/test-images.md).
