"""Kernel-level equivalence tests for converted Wan 2.1 layers.

Compares `SVDQW4A4Linear` outputs against a fake-quant reference computed from
the deepcompressor checkpoint (`model.pt` + `smooth.pt` + `branch.pt`) on
identical random inputs. The reference simulates the kernel's dynamic
activation quantization (per-token groups, absmax/7); an encoding mismatch
(sint4 decoded as fp4 or vice versa) fails these thresholds instantly.

The shifted unsigned layer (`ffn.net.2`, the post-GELU projection) uses a
looser bound: the exact unsigned rounding-with-shift semantics live inside the
fused quantize kernel and are only approximated by the reference.

Requires two environment variables:
- ``NUNCHAKU_TEST_WAN_CKPT``: the converted single-file safetensors checkpoint
- ``NUNCHAKU_TEST_WAN_QUANT_DIR``: the deepcompressor run's model directory
  (containing ``model.pt``, ``smooth.pt``, ``branch.pt``)
"""

import os

import pytest
import torch

pytest.importorskip("nunchaku._C", reason="requires the compiled nunchaku extension")

from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.models.transformers.transformer_wan import NunchakuWanTransformer3DModel

CKPT = os.environ.get("NUNCHAKU_TEST_WAN_CKPT", "")
QUANT_DIR = os.environ.get("NUNCHAKU_TEST_WAN_QUANT_DIR", "")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        not (CKPT and QUANT_DIR), reason="set NUNCHAKU_TEST_WAN_CKPT and NUNCHAKU_TEST_WAN_QUANT_DIR"
    ),
]

# (converted module, source members, is_shifted_unsigned)
SAMPLED_LAYERS = [
    ("blocks.0.attn1.to_qkv", ["blocks.0.attn1.to_q", "blocks.0.attn1.to_k", "blocks.0.attn1.to_v"], False),
    ("blocks.0.attn2.to_kv", ["blocks.0.attn2.to_k", "blocks.0.attn2.to_v"], False),
    ("blocks.0.attn1.to_out.0", ["blocks.0.attn1.to_out.0"], False),
    ("blocks.0.ffn.net.0.proj", ["blocks.0.ffn.net.0.proj"], False),
    ("blocks.0.ffn.net.2", ["blocks.0.ffn.net.2"], True),
    ("blocks.29.attn1.to_qkv", ["blocks.29.attn1.to_q", "blocks.29.attn1.to_k", "blocks.29.attn1.to_v"], False),
    ("blocks.29.ffn.net.0.proj", ["blocks.29.ffn.net.0.proj"], False),
]

# thresholds validated on known-good layers of the first calibrated 1.3B INT4
# checkpoint (cosine 0.9995-0.9997, rel MSE 0.7-1.1e-3); an encoding or format
# mismatch lands orders of magnitude outside them
COSINE_THRESHOLD = 0.999
REL_MSE_THRESHOLD = 2e-3
SHIFTED_COSINE_THRESHOLD = 0.98
SHIFTED_REL_MSE_THRESHOLD = 5e-2


@pytest.fixture(scope="module")
def transformer():
    return NunchakuWanTransformer3DModel.from_pretrained(CKPT, device="cuda")


@pytest.fixture(scope="module")
def reference_dicts():
    state_dict = torch.load(os.path.join(QUANT_DIR, "model.pt"), map_location="cpu")
    smooth_dict = torch.load(os.path.join(QUANT_DIR, "smooth.pt"), map_location="cpu")
    branch_path = os.path.join(QUANT_DIR, "branch.pt")
    branch_dict = torch.load(branch_path, map_location="cpu") if os.path.exists(branch_path) else {}
    return state_dict, smooth_dict, branch_dict


def fake_quantize_activations(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    grouped = x.unflatten(-1, (-1, group_size))
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7.0
    return (grouped / scale).round().clamp(-8, 7).mul(scale).flatten(-2)


def fake_quantize_activations_unsigned(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    grouped = x.unflatten(-1, (-1, group_size))
    scale = grouped.amax(dim=-1, keepdim=True).clamp(min=1e-12) / 15.0
    return (grouped / scale).round().clamp(0, 15).mul(scale).flatten(-2)


def resolve_member(state_dict, name):
    # shifted linears store their weights under `<name>.linear`
    if f"{name}.weight" not in state_dict and f"{name}.linear.weight" in state_dict:
        return f"{name}.linear"
    return name


@pytest.mark.parametrize(
    "converted_name,member_names,shifted_unsigned",
    SAMPLED_LAYERS,
    ids=[n for n, _, _ in SAMPLED_LAYERS],
)
def test_linear_equivalence(transformer, reference_dicts, converted_name, member_names, shifted_unsigned):
    state_dict, smooth_dict, branch_dict = reference_dicts
    module = transformer.get_submodule(converted_name)
    assert isinstance(module, SVDQW4A4Linear)
    member_names = [resolve_member(state_dict, name) for name in member_names]
    anchor_name = member_names[0]

    weight = torch.cat([state_dict[f"{name}.weight"] for name in member_names], dim=0).double()
    bias = torch.cat([state_dict[f"{name}.bias"] for name in member_names], dim=0).double()
    smooth = smooth_dict[anchor_name].double()
    branch = branch_dict.get(anchor_name, None)

    torch.manual_seed(0)
    if shifted_unsigned:
        # the post-GELU input distribution: mostly positive with a small negative tail
        x = torch.nn.functional.gelu(
            torch.randn(1, 1024, module.in_features, dtype=torch.float32, device="cuda") * 1.5,
            approximate="tanh",
        ).to(torch.bfloat16)
        shift = state_dict[f"{anchor_name[: -len('.linear')]}.shift"].double()
    else:
        x = torch.randn(1, 1024, module.in_features, dtype=torch.bfloat16, device="cuda") * 0.5
        shift = None

    y_kernel = module(x).float().cpu().double().flatten()

    x_smoothed = x.cpu().double() / smooth
    if shifted_unsigned:
        x_effective = x_smoothed + (shift / smooth if shift.numel() > 1 else shift)
        x_quantized = fake_quantize_activations_unsigned((x.cpu().double() + shift) / smooth)
    else:
        x_effective = x_smoothed
        x_quantized = fake_quantize_activations(x_smoothed)
    y_ref = x_quantized @ weight.T + bias
    if branch is not None:
        y_ref = y_ref + x_effective @ branch["a.weight"].double().T @ branch["b.weight"].double().T
    y_ref = y_ref.flatten()

    cosine = torch.nn.functional.cosine_similarity(y_kernel, y_ref, dim=0)
    rel_mse = ((y_kernel - y_ref) ** 2).mean() / (y_ref**2).mean().clamp(min=1e-12)
    cos_thresh = SHIFTED_COSINE_THRESHOLD if shifted_unsigned else COSINE_THRESHOLD
    mse_thresh = SHIFTED_REL_MSE_THRESHOLD if shifted_unsigned else REL_MSE_THRESHOLD
    assert cosine > cos_thresh, f"{converted_name}: cosine {cosine:.6f}"
    assert rel_mse < mse_thresh, f"{converted_name}: rel MSE {rel_mse:.3e}"
