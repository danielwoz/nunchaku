"""Kernel-level equivalence tests for converted Wan 2.1 layers.

Compares `SVDQW4A4Linear` outputs against a fake-quant reference computed from
the deepcompressor checkpoint (`model.pt` + `smooth.pt` + `branch.pt`) on
identical random inputs. An encoding mismatch (sint4 decoded as fp4 or vice
versa) fails these thresholds instantly.

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

# sampled layers: qkv, out, ffn up/down in the first and last blocks
SAMPLED_LAYERS = [
    ("blocks.0.attn1.to_qkv", ["blocks.0.attn1.to_q", "blocks.0.attn1.to_k", "blocks.0.attn1.to_v"]),
    ("blocks.0.attn2.to_kv", ["blocks.0.attn2.to_k", "blocks.0.attn2.to_v"]),
    ("blocks.0.ffn.net.0.proj", ["blocks.0.ffn.net.0.proj"]),
    ("blocks.0.ffn.net.2", ["blocks.0.ffn.net.2"]),
    ("blocks.0.attn1.to_out.0", ["blocks.0.attn1.to_out.0"]),
]

COSINE_THRESHOLD = 0.999
REL_MSE_THRESHOLD = 1e-3


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


def reference_forward(x, state_dict, smooth_dict, branch_dict, member_names, anchor_name):
    """Fake-quant reference: smoothed input through the fake-quantized weights + low-rank branch."""
    weight = torch.cat([state_dict[f"{name}.weight"] for name in member_names], dim=0)
    bias = torch.cat([state_dict[f"{name}.bias"] for name in member_names], dim=0)
    smooth = smooth_dict[anchor_name]
    x_s = (x.double() / smooth.double())
    y = x_s @ weight.double().T + bias.double()
    branch = branch_dict.get(anchor_name, None)
    if branch is not None:
        y = y + x_s @ branch["a.weight"].double().T @ branch["b.weight"].double().T
    return y


@pytest.mark.parametrize("converted_name,member_names", SAMPLED_LAYERS, ids=[n for n, _ in SAMPLED_LAYERS])
def test_linear_equivalence(transformer, reference_dicts, converted_name, member_names):
    state_dict, smooth_dict, branch_dict = reference_dicts
    module = transformer.get_submodule(converted_name)
    assert isinstance(module, SVDQW4A4Linear)
    anchor_name = smooth_dict and member_names[0]
    # the ffn down-proj smooth cache key follows the shifted-linear rname when present
    if anchor_name not in smooth_dict and f"{anchor_name}.linear" in smooth_dict:
        anchor_name = f"{anchor_name}.linear"
    torch.manual_seed(0)
    x = torch.randn(1, 1024, module.in_features, dtype=torch.bfloat16, device="cuda") * 0.5
    y_kernel = module(x).float().cpu().double().flatten()
    y_ref = reference_forward(
        x.cpu(), state_dict, smooth_dict, branch_dict, member_names, anchor_name
    ).flatten()
    cosine = torch.nn.functional.cosine_similarity(y_kernel, y_ref, dim=0)
    rel_mse = ((y_kernel - y_ref) ** 2).mean() / (y_ref**2).mean().clamp(min=1e-12)
    assert cosine > COSINE_THRESHOLD, f"{converted_name}: cosine {cosine:.6f}"
    assert rel_mse < REL_MSE_THRESHOLD, f"{converted_name}: rel MSE {rel_mse:.3e}"
