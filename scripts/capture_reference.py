"""Capture synthetic upstream component inputs, weights, and outputs on CPU."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save

from idol.camera import decode_camera
from idol.models.layers import RePo3DRotaryEmbedding, RopeMultiHeadAttention
from idol.sampler import sample_flow


def capture():
    """Return a self-contained synthetic reference for selected components."""
    torch.manual_seed(1729)
    width, heads = 24, 2
    positions = RePo3DRotaryEmbedding(width, num_heads=heads, head_dim=width // heads)
    positions.real_valued = True
    attention = RopeMultiHeadAttention(width, num_heads=heads)
    positions.eval()
    attention.eval()
    features = torch.linspace(-1.0, 1.0, steps=width, dtype=torch.float32)
    x = torch.stack((features, features.square(), torch.sin(features * 3.0))).unsqueeze(0)
    noise = torch.tensor([[10.0]], dtype=torch.float32)
    condition = torch.tensor([[2.0]], dtype=torch.float32)
    negative_condition = torch.tensor([[1.0]], dtype=torch.float32)
    camera = torch.tensor([0.0, 1.0, 0.0, 0.5, 0.25], dtype=torch.float32)

    def velocity(sample, time, cond):
        return cond

    with torch.no_grad():
        rotary = positions(x)
        attention_output = attention(x, rotary)
        sampled = sample_flow(velocity, noise, condition, negative_condition,
                              steps=2, rescale_t=3.0, cfg_strength=3.0, verbose=False)
        decoded_camera = decode_camera(camera)
    tensors = {
        "repo.input": x,
        "repo.output": rotary,
        "attention.input": x,
        "attention.rotary": rotary,
        "attention.output": attention_output,
        "sampler.noise": noise,
        "sampler.condition": condition,
        "sampler.negative_condition": negative_condition,
        "sampler.output": sampled,
        "camera.input": camera,
        "camera.output": torch.tensor([decoded_camera[key] for key in ("yaw", "pitch", "radius", "fov")],
                                      dtype=torch.float64),
    }
    for prefix, module in (("repo", positions), ("attention", attention)):
        tensors.update({f"{prefix}.weights.{name}": value.detach().cpu().contiguous()
                        for name, value in module.state_dict().items()})
    return {name: tensor.detach().cpu().contiguous().clone() for name, tensor in tensors.items()}


def write_capture(destination):
    """Write only synthetic tensors, refusing to replace an existing file."""
    payload = save(capture(), metadata={"kind": "synthetic-component-reference", "version": "1"})
    with Path(destination).open("xb") as file:
        file.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new safetensors file")
    args = parser.parse_args()
    write_capture(args.output)
    print("Synthetic component reference written.")


if __name__ == "__main__":
    main()
