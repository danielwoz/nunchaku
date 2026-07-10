"""Structural tests for the Wan 2.1 loader: checkpoint schema round-trip and arch gating."""

import json

import pytest
import torch

pytest.importorskip("nunchaku._C", reason="requires the compiled nunchaku extension")

from diffusers import WanTransformer3DModel
from safetensors.torch import save_file

from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.models.transformers.transformer_wan import (
    NunchakuWanTransformer3DModel,
    NunchakuWanTransformerBlock,
)

TINY_CONFIG = {
    "patch_size": [1, 2, 2],
    "num_attention_heads": 2,
    "attention_head_dim": 64,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 32,
    "freq_dim": 256,
    "ffn_dim": 256,
    "num_layers": 2,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 32,
}


def build_quantization_config(precision: str, rank: int = 32) -> dict:
    fp4 = precision == "nvfp4"
    return {
        "method": "svdquant",
        "weight": {
            "dtype": "fp4_e2m1_all" if fp4 else "int4",
            "scale_dtype": [None, "fp8_e4m3_nan"] if fp4 else None,
            "group_size": 16 if fp4 else 64,
        },
        "activation": {
            "dtype": "fp4_e2m1_all" if fp4 else "int4",
            "scale_dtype": "fp8_e4m3_nan" if fp4 else None,
            "group_size": 16 if fp4 else 64,
        },
        "rank": rank,
    }


def make_checkpoint(tmp_path, precision: str):
    """Build a random-weight single-file checkpoint with the runtime's exact key schema."""
    with torch.device("meta"):
        transformer = NunchakuWanTransformer3DModel.from_config(TINY_CONFIG).to(torch.bfloat16)
    transformer._patch_model(precision=precision, rank=32)
    transformer = transformer.to_empty(device="cpu")
    state_dict = {}
    generator = torch.Generator().manual_seed(0)
    for key, value in transformer.state_dict().items():
        if value.dtype in (torch.int8,):
            state_dict[key] = torch.randint(-128, 127, value.shape, dtype=torch.int8, generator=generator)
        elif value.dtype == torch.float8_e4m3fn:
            state_dict[key] = torch.rand(value.shape, generator=generator).to(torch.float8_e4m3fn)
        else:
            state_dict[key] = torch.rand(value.shape, generator=generator).to(value.dtype)
    metadata = {
        "config": json.dumps(TINY_CONFIG),
        "model_class": "NunchakuWanTransformer3DModel",
        "quantization_config": json.dumps(build_quantization_config(precision)),
    }
    path = tmp_path / f"wan-tiny-svdq-{precision}.safetensors"
    save_file(state_dict, path, metadata=metadata)
    return path, state_dict


def local_precision() -> str:
    capability = torch.cuda.get_device_capability(0)
    return "nvfp4" if f"{capability[0]}{capability[1]}" in ("120", "121") else "int4"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_wan_from_pretrained_round_trip(tmp_path):
    precision = local_precision()
    path, state_dict = make_checkpoint(tmp_path, precision)
    transformer = NunchakuWanTransformer3DModel.from_pretrained(path, device="cuda")
    assert isinstance(transformer, NunchakuWanTransformer3DModel)
    for block in transformer.blocks:
        assert isinstance(block, NunchakuWanTransformerBlock)
        assert isinstance(block.attn1.to_qkv, SVDQW4A4Linear)
        assert isinstance(block.attn2.to_q, SVDQW4A4Linear)
        assert isinstance(block.attn2.to_kv, SVDQW4A4Linear)
        assert block.attn1.to_qkv.precision == precision
    # rope buffers must be re-created after to_empty
    assert transformer.rope.freqs_cos.isfinite().all()
    # loaded tensors must round-trip exactly
    loaded = transformer.state_dict()
    for key, value in state_dict.items():
        if key.endswith(".wtscale"):
            continue
        assert torch.equal(loaded[key].cpu(), value), f"mismatch for {key}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_wan_wrong_pairing_fails(tmp_path):
    # loading a checkpoint whose encoding mismatches the GPU arch must raise,
    # never silently decode the wrong 4-bit format
    wrong_precision = "int4" if local_precision() == "nvfp4" else "nvfp4"
    path, _ = make_checkpoint(tmp_path, wrong_precision)
    with pytest.raises(ValueError):
        NunchakuWanTransformer3DModel.from_pretrained(path, device="cuda")
