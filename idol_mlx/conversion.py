"""Explicit, lossless tensor-name and standard convolution layout mapping."""

from dataclasses import dataclass

from .weights import CheckpointError, validate_tensors


@dataclass(frozen=True)
class TensorMapping:
    """Map one input tensor to one output using an explicitly selected layout."""

    source: str
    target: str
    layout: str = "identity"

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty tensor name.")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("target must be a non-empty tensor name.")
        if self.layout not in ("identity", "conv2d_oihw_to_ohwi"):
            raise ValueError("Unsupported tensor layout conversion.")


def convert_tensors(tensors, mappings, expected):
    """Convert a complete MLX-array dictionary using explicit one-to-one rules.

    Identity preserves values and layout. Standard 2-D convolution maps OIHW
    to OHWI. The caller is responsible for selecting an applicable operation;
    no architecture or grouped-convolution compatibility is inferred.
    """
    if not isinstance(tensors, dict) or not tensors or any(
        not isinstance(name, str) or not name for name in tensors
    ):
        raise ValueError("Source tensors must be a non-empty dictionary with string names.")
    if not isinstance(mappings, (list, tuple)) or not mappings or any(
        not isinstance(rule, TensorMapping) for rule in mappings
    ):
        raise ValueError("Mappings must be a non-empty sequence of TensorMapping entries.")
    sources = [rule.source for rule in mappings]
    targets = [rule.target for rule in mappings]
    if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
        raise ValueError("Mapping sources and targets must each be unique.")
    missing = sorted(set(sources) - tensors.keys())
    unexpected = sorted(tensors.keys() - set(sources))
    if missing or unexpected:
        raise CheckpointError(f"Source keys do not match mapping: missing={missing}, unexpected={unexpected}.")
    converted = {}
    for rule in mappings:
        value = tensors[rule.source]
        if rule.layout == "conv2d_oihw_to_ohwi":
            if len(value.shape) != 4:
                raise CheckpointError(f"Expected a rank-4 Conv2d tensor for {rule.source}.")
            value = value.transpose(0, 2, 3, 1)
        converted[rule.target] = value
    return validate_tensors(converted, expected)
