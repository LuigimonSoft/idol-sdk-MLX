"""Read and validate the shared structure of an Idol model configuration."""

import json
from pathlib import Path


class ConfigError(ValueError):
    """The model configuration cannot be read or has an invalid structure."""


def load_config(path):
    """Return a validated config dictionary without modifying unknown fields.

    Individual model arguments require additional model-specific validation.
    The path is selected by a trusted local caller.
    """
    try:
        with Path(path).open(encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigError("Cannot read model configuration as JSON.") from error
    if not isinstance(config, dict):
        raise ConfigError("Model configuration must be a JSON object.")
    for name in ("denoiser", "decoder", "decoder_gs"):
        section = config.get(name)
        if not isinstance(section, dict):
            raise ConfigError(f"{name} must be an object.")
        if not isinstance(section.get("name"), str) or not section["name"].strip():
            raise ConfigError(f"{name}.name must be a non-empty string.")
        if not isinstance(section.get("args"), dict):
            raise ConfigError(f"{name}.args must be an object.")
    for name in ("image_size", "max_voxel_level"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"{name} must be a positive integer.")
    encoders = config.get("image_encoders")
    if not isinstance(encoders, dict):
        raise ConfigError("image_encoders must be an object.")
    for name in ("dinov3", "flux2_vae"):
        value = encoders.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"image_encoders.{name} must be a non-empty string.")
    return config
