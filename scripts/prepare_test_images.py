"""Prepare a small, reproducible set of public and generated test images."""

import argparse
import hashlib
import os
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from PIL import Image, ImageDraw

PHOTO_BASE = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/"
PHOTOS = {
    "coffee.png": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
    "clock_motion.png": "f029226b28b642e80113d86622e9b215ee067a0966feaf5e60604a1e05733955",
}


def checked_photo(name, expected_hash):
    """Read the pinned public image and reject changed or corrupted content."""
    with urlopen(PHOTO_BASE + name, timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {name}")
    return data


def synthetic_images():
    """Draw a simple opaque object and a separate transparent object."""
    opaque = Image.new("RGB", (256, 256), (225, 235, 245))
    draw = ImageDraw.Draw(opaque)
    draw.rectangle((78, 66, 177, 202), fill=(70, 125, 205))
    draw.ellipse((78, 50, 177, 82), fill=(105, 165, 235))
    transparent = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(transparent)
    draw.ellipse((48, 48, 207, 207), fill=(230, 100, 45, 255))
    draw.ellipse((82, 75, 135, 125), fill=(255, 170, 85, 255))
    return {"synthetic_opaque.png": opaque, "synthetic_alpha.png": transparent}


def prepare(output):
    """Write fixtures into a directory chosen by a trusted local caller."""
    output = Path(output)
    names = [*PHOTOS, "synthetic_opaque.png", "synthetic_alpha.png"]
    if any(os.path.lexists(output / name) for name in names):
        raise FileExistsError("Output already contains a fixture; use an empty directory")
    photos = {name: checked_photo(name, digest) for name, digest in PHOTOS.items()}
    images = {}
    for name, image in synthetic_images().items():
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        images[name] = buffer.getvalue()
    output.mkdir(parents=True, exist_ok=True)
    for name, data in {**photos, **images}.items():
        with (output / name).open("xb") as file:
            file.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="trusted empty destination directory outside the repository")
    args = parser.parse_args()
    prepare(args.output)
    print(f"Prepared {len(PHOTOS) + 2} test images.")


if __name__ == "__main__":
    main()
