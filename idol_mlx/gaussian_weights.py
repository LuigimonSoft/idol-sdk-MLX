"""Strict FP32 checkpoint loading for the preliminary Gaussian decoder."""

import re

from mlx.utils import tree_flatten

from .conversion import TensorMapping, convert_tensors
from .gaussian_decoder import GaussianFixedlenDecoder
from .weights import TensorSpec, load_weights


_MLP_NAME = re.compile(r"^(blocks\.\d+\.mlp\.mlp)\.layers\.(\d+)\.")


def load_gaussian_checkpoint(model, path):
    """Load an upstream-named FP32 checkpoint including the two model buffers.

    Source and target names, shapes and dtypes are validated before mutation.
    No weights are cast or ignored.
    """
    if not isinstance(model, GaussianFixedlenDecoder):
        raise TypeError("model must be a GaussianFixedlenDecoder.")
    target = dict(tree_flatten(model.parameters()))
    expected = {name: TensorSpec(tuple(value.shape), str(value.dtype).removeprefix("mlx.core."))
                for name, value in target.items()}
    if any(spec.dtype != "float32" for spec in expected.values()):
        raise ValueError("All target model tensors must be float32.")
    mappings = []
    source_schema = {}
    for target_name, spec in expected.items():
        source_name = _MLP_NAME.sub(r"\1.\2.", target_name)
        if source_name in source_schema:
            raise ValueError("Gaussian mapping contains duplicate source names.")
        source_schema[source_name] = spec
        mappings.append(TensorMapping(source_name, target_name))
    source = load_weights(path, source_schema)
    converted = convert_tensors(source, mappings, expected)
    model.load_weights(list(converted.items()), strict=True)
