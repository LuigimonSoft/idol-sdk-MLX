"""Strict FP32 denoiser checkpoint name mapping for the preliminary MLX model."""

import re

from mlx.utils import tree_flatten

from .conversion import TensorMapping, convert_tensors
from .dit import LatentSeqMMFlowModel
from .weights import TensorSpec, load_weights


# Only sequential containers differ between the pinned PyTorch and MLX models.
_SEQUENTIAL_NAMES = (
    (re.compile(r"^adaLN_modulation\.layers\.(\d+)\."), r"adaLN_modulation.\1."),
    (re.compile(r"^t_embedder\.mlp\.layers\.(\d+)\."), r"t_embedder.mlp.\1."),
    (re.compile(r"^cam_refiner\.mlp\.layers\.(\d+)\."), r"cam_refiner.mlp.\1."),
    (re.compile(r"^((?:noise_refiner|context_refiner|blocks)\.\d+\.mlp\.mlp)\.layers\.(\d+)\."), r"\1.\2."),
)


def load_dit_checkpoint(model, path):
    """Load a trusted local PyTorch-named FP32 safetensors checkpoint strictly.

    The schema comes from the supplied model; no tensors are cast, reshaped or
    ignored. Validate every source and target before updating model parameters.
    The mapping is validated on synthetic configurations, not trained weights.
    """
    if not isinstance(model, LatentSeqMMFlowModel):
        raise TypeError("model must be a LatentSeqMMFlowModel.")
    target = dict(tree_flatten(model.parameters()))
    expected = {name: TensorSpec(tuple(value.shape), str(value.dtype).removeprefix("mlx.core."))
                for name, value in target.items()}
    if any(spec.dtype != "float32" for spec in expected.values()):
        raise ValueError("All target model tensors must be float32.")
    mappings = []
    source_schema = {}
    for target_name, spec in expected.items():
        source_name = target_name
        for pattern, replacement in _SEQUENTIAL_NAMES:
            source_name = pattern.sub(replacement, source_name)
        if source_name in source_schema:
            raise ValueError("Denoiser mapping contains duplicate source names.")
        source_schema[source_name] = spec
        mappings.append(TensorMapping(source_name, target_name))
    source = load_weights(path, source_schema)
    converted = convert_tensors(source, mappings, expected)
    model.load_weights(list(converted.items()), strict=True)
