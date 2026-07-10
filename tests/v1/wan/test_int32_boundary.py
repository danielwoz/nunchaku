"""Shape/boundary sweep for the W4A4 kernels at Wan 2.1-14B GEMM shapes.

Token counts cover 480P/81f (32768), 720P/81f (75776), 720P/121f (111616),
and 720P/161f (147712, which overflows the pre-fix int32 M*N offsets).
Asserts the kernels neither abort nor write garbage at these sizes.
"""

import gc

import pytest
import torch

pytest.importorskip("nunchaku._C", reason="requires the compiled nunchaku extension")

from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.utils import get_precision

TOKEN_COUNTS = [256, 32768, 75776, 111616, 147712]
# (in_features, out_features): 14B fused qkv and ffn up projection
LAYER_SHAPES = [(5120, 15360), (5120, 13824)]


def make_linear(in_features: int, out_features: int) -> SVDQW4A4Linear:
    precision = "nvfp4" if get_precision() == "fp4" else "int4"
    linear = SVDQW4A4Linear(
        in_features=in_features,
        out_features=out_features,
        rank=32,
        precision=precision,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    with torch.no_grad():
        linear.qweight.zero_()
        linear.wscales.copy_(torch.full_like(linear.wscales.to(torch.float32), 0.01).to(linear.wscales.dtype))
        if linear.bias is not None:
            linear.bias.fill_(0.5)
        linear.smooth_factor.fill_(1.0)
        linear.smooth_factor_orig.fill_(1.0)
        linear.proj_down.normal_(std=0.01)
        linear.proj_up.zero_()
        if linear.wcscales is not None:
            linear.wcscales.fill_(1.0)
    return linear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("in_features,out_features", LAYER_SHAPES, ids=["qkv-14b", "ffn-up-14b"])
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
def test_no_int32_wrap(num_tokens, in_features, out_features):
    gc.collect()
    torch.cuda.empty_cache()
    required_gib = (num_tokens * (in_features + out_features) * 2 * 3) / 2**30
    free_gib = torch.cuda.mem_get_info()[0] / 2**30
    if free_gib < required_gib + 4:
        pytest.skip(f"needs ~{required_gib:.0f} GiB free VRAM")
    linear = make_linear(in_features, out_features)
    x = torch.randn(1, num_tokens, in_features, dtype=torch.bfloat16, device="cuda")
    y = linear(x)
    torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()
    assert y.shape == (1, num_tokens, out_features)
    # zero qweight + zero proj_up + bias 0.5 => every output must be exactly 0.5;
    # an int32 offset wrap leaves rows unwritten or writes out of bounds.
    # checked in chunks: a full float copy of the output would not fit on 24 GB
    max_dev = 0.0
    for chunk in y.view(-1, out_features).split(65536):
        assert chunk.isfinite().all()
        max_dev = max(max_dev, float((chunk.float() - 0.5).abs().max()))
    assert max_dev < 1e-3
