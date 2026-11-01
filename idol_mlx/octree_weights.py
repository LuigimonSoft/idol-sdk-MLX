"""Strict FP32 checkpoint loading for the preliminary octree decoder."""

import re

from mlx.utils import tree_flatten

from .conversion import TensorMapping, convert_tensors
from .octree import OctreeProbabilityFixedlenDecoder
from .weights import TensorSpec, load_weights


_SEQUENTIAL_NAMES = (
    (re.compile(r"^adaLN_modulation\.layers\.(\d+)\."), r"adaLN_modulation.\1."),
    (re.compile(r"^l_embedder\.mlp\.layers\.(\d+)\."), r"l_embedder.mlp.\1."),
    (re.compile(r"^(blocks\.\d+\.mlp\.mlp)\.layers\.(\d+)\."), r"\1.\2."),
)


def load_octree_checkpoint(model, path):
    """Load an upstream-named FP32 safetensors checkpoint without casting.

    The complete source and target schemas are checked before changing the model.
    """
    if not isinstance(model, OctreeProbabilityFixedlenDecoder):
        raise TypeError("model must be an OctreeProbabilityFixedlenDecoder.")
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
            raise ValueError("Octree mapping contains duplicate source names.")
        source_schema[source_name] = spec
        mappings.append(TensorMapping(source_name, target_name))
    source = load_weights(path, source_schema)
    converted = convert_tensors(source, mappings, expected)
    model.load_weights(list(converted.items()), strict=True)
