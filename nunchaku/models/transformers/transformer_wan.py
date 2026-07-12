"""
This module provides the implementation of NunchakuWanTransformer3DModel and its building blocks.
"""

import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanRotaryPosEmbed,
    WanTransformer3DModel,
    WanTransformerBlock,
)
from diffusers.utils import logging as diffusers_logging
from huggingface_hub import utils

from ...utils import check_hardware_compatibility, get_precision_from_quantization_config
from ..attention import NunchakuBaseAttention, NunchakuFeedForward
from ..attention_processors.wan import NunchakuWanNaiveFA2Processor
from ..linear import SVDQW4A4Linear
from ..utils import CPUOffloadManager, fuse_linears
from .utils import NunchakuModelLoaderMixin, patch_scale_key

logger = diffusers_logging.get_logger(__name__)


class NunchakuWanAttention(NunchakuBaseAttention):
    """
    Nunchaku-optimized quantized attention module for Wan 2.1.

    Self-attention fuses ``to_q``/``to_k``/``to_v`` into a single quantized
    ``to_qkv``; cross-attention keeps a quantized ``to_q`` and fuses the text
    K/V projections into a quantized ``to_kv``. The I2V image-stream
    projections (``add_k_proj``/``add_v_proj``, ~257 CLIP tokens per video)
    stay unquantized.

    Parameters
    ----------
    other : WanAttention
        The original Wan attention module to wrap and quantize.
    processor : str, default="flashattn2"
        The attention processor to use.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, other: WanAttention, processor: str = "flashattn2", skips: set[str] | None = None, **kwargs):
        super(NunchakuWanAttention, self).__init__(processor)
        skips = skips or set()
        self.inner_dim = other.inner_dim
        self.kv_inner_dim = other.kv_inner_dim
        self.heads = other.heads
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.cross_attention_dim_head = other.cross_attention_dim_head
        self.is_cross_attention = other.is_cross_attention

        self.norm_q = other.norm_q
        self.norm_k = other.norm_k

        # skipped units keep the original bf16 linears; the processor falls
        # back to unfused projections when the fused module is absent
        if other.is_cross_attention:
            if "to_q" in skips:
                self.to_q = other.to_q
            else:
                self.to_q = SVDQW4A4Linear.from_linear(other.to_q, **kwargs)
            if "to_kv" in skips:
                self.to_k = other.to_k
                self.to_v = other.to_v
            else:
                with torch.device("meta"):
                    to_kv = fuse_linears([other.to_k, other.to_v])
                self.to_kv = SVDQW4A4Linear.from_linear(to_kv, **kwargs)
        else:
            if "to_qkv" in skips:
                self.to_q = other.to_q
                self.to_k = other.to_k
                self.to_v = other.to_v
            else:
                with torch.device("meta"):
                    to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
                self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        self.to_out = other.to_out
        if "to_out.0" not in skips:
            self.to_out[0] = SVDQW4A4Linear.from_linear(other.to_out[0], **kwargs)

        self.add_k_proj = other.add_k_proj
        self.add_v_proj = other.add_v_proj
        if other.added_kv_proj_dim is not None:
            self.norm_added_k = other.norm_added_k

    def set_processor(self, processor: str):
        """
        Set the attention processor.

        Parameters
        ----------
        processor : str
            Name of the processor to use. Only "flashattn2" is supported for now.
            See :class:`~nunchaku.models.attention_processors.wan.NunchakuWanNaiveFA2Processor`.

        Raises
        ------
        ValueError
            If the processor is not supported.
        """
        if processor == "flashattn2":
            self.processor = NunchakuWanNaiveFA2Processor()
        else:
            raise ValueError(f"Processor {processor} is not supported")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)


class NunchakuWanTransformerBlock(WanTransformerBlock):
    """
    Quantized Wan 2.1 transformer block.

    The FP32 layer norms, the per-block ``scale_shift_table``, and the
    across-heads QK RMSNorms stay unquantized; there is no per-block modulation
    linear to quantize (Wan's adaLN projection is model-level and skipped).

    Parameters
    ----------
    other : WanTransformerBlock
        The original transformer block to wrap and quantize.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, other: WanTransformerBlock, skips: set[str] | None = None, **kwargs):
        super(WanTransformerBlock, self).__init__()
        skips = skips or set()
        self.norm1 = other.norm1
        self.attn1 = NunchakuWanAttention(
            other.attn1, skips={s[len("attn1.") :] for s in skips if s.startswith("attn1.")}, **kwargs
        )
        self.attn2 = NunchakuWanAttention(
            other.attn2, skips={s[len("attn2.") :] for s in skips if s.startswith("attn2.")}, **kwargs
        )
        self.norm2 = other.norm2
        skip_up, skip_down = "ffn.net.0.proj" in skips, "ffn.net.2" in skips
        if not skip_up and not skip_down:
            self.ffn = NunchakuFeedForward(other.ffn, **kwargs)
        else:
            # partially skipped: keep the diffusers FeedForward (sequential
            # forward) and quantize only the non-skipped projection
            self.ffn = other.ffn
            if not skip_up:
                self.ffn.net[0].proj = SVDQW4A4Linear.from_linear(self.ffn.net[0].proj, **kwargs)
            if not skip_down:
                self.ffn.net[2] = SVDQW4A4Linear.from_linear(self.ffn.net[2], **kwargs)
                self.ffn.net[2].act_unsigned = self.ffn.net[2].precision != "nvfp4"
        self.norm3 = other.norm3
        self.scale_shift_table = other.scale_shift_table


class NunchakuWanTransformer3DModel(WanTransformer3DModel, NunchakuModelLoaderMixin):
    """
    Quantized Wan 2.1 Transformer3DModel.

    Supports quantized transformer blocks and optional per-block CPU offloading
    (which makes the 14B model usable on 24 GB GPUs).

    Attributes
    ----------
    offload : bool
        Whether CPU offloading is enabled.
    offload_manager : CPUOffloadManager or None
        Manager for offloading transformer blocks.
    _is_initialized : bool
        Whether the model has been patched for quantization.
    """

    # the explicit signature (mirroring `WanTransformer3DModel.__init__`) is
    # required: diffusers' `from_config` reads the `__init__` signature to
    # decide which config keys to apply, and a bare `(*args, **kwargs)`
    # signature would silently fall back to the 14B default dimensions
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        offload: bool = False,
    ):
        self.offload = offload
        self.offload_manager = None
        self._is_initialized = False
        super().__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
        )

    def _patch_model(self, skips: list[str] | None = None, **kwargs):
        """
        Patch the transformer blocks for quantization.

        Parameters
        ----------
        skips : list of str, optional
            Units to keep unquantized, either globally (``"attn2.to_kv"``) or
            per block (``"blocks.5.attn2.to_kv"``).
        **kwargs
            Additional arguments for quantization (e.g. ``precision``, ``rank``).

        Returns
        -------
        self
        """
        skips = skips or []
        for i, block in enumerate(self.blocks):
            block_skips = set()
            for skip in skips:
                if skip.startswith("blocks."):
                    prefix = f"blocks.{i}."
                    if skip.startswith(prefix):
                        block_skips.add(skip[len(prefix) :])
                else:  # a bare unit name applies to every block
                    block_skips.add(skip)
            self.blocks[i] = NunchakuWanTransformerBlock(block, skips=block_skips, **kwargs)
        self._is_initialized = True
        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a quantized Wan model from a single-file checkpoint.

        The checkpoint's ``quantization_config`` metadata determines the weight
        encoding (int4 vs fp4-e2m1), and :func:`check_hardware_compatibility`
        raises if the encoding does not match the GPU architecture — an fp4
        checkpoint never silently decodes as int4 (or vice versa).

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the checkpoint. It can be a local file or a remote HuggingFace path.
        **kwargs
            Additional arguments for loading and quantization
            (e.g. ``device``, ``torch_dtype``, ``offload``).

        Returns
        -------
        NunchakuWanTransformer3DModel
            The loaded and quantized model.
        """
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        offload = kwargs.get("offload", False)
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        config = json.loads(metadata.get("config", "{}"))
        rank = quantization_config.get("rank", 32)
        transformer = transformer.to(torch_dtype)

        # the precision comes from the checkpoint metadata, never from the local
        # arch; the compatibility check hard-fails on a wrong arch/encoding pair
        precision = get_precision_from_quantization_config(quantization_config)
        if torch.cuda.is_available():
            gate_device = device if torch.device(device).type == "cuda" else "cuda"
            check_hardware_compatibility(quantization_config, gate_device)
        transformer._patch_model(
            precision=precision, rank=rank, skips=quantization_config.get("skips", None)
        )

        transformer = transformer.to_empty(device=device)
        # re-create the rotary embedding: its frequency buffers are
        # non-persistent, so `to_empty` leaves them uninitialized
        transformer.rope = WanRotaryPosEmbed(
            attention_head_dim=config.get("attention_head_dim", 128),
            patch_size=tuple(config.get("patch_size", (1, 2, 2))),
            max_seq_len=config.get("rope_max_seq_len", 1024),
        ).to(device)

        patch_scale_key(transformer, model_state_dict)
        for module in transformer.modules():
            if isinstance(module, SVDQW4A4Linear) and isinstance(module.wtscale, torch.Tensor):
                module.wtscale = float(module.wtscale.item())

        transformer.load_state_dict(model_state_dict)
        transformer.set_offload(offload)

        return transformer

    def set_offload(self, offload: bool, **kwargs):
        """
        Enable or disable asynchronous CPU offloading for transformer blocks.

        Parameters
        ----------
        offload : bool
            Whether to enable offloading.
        **kwargs
            Additional arguments for the offload manager.

        See Also
        --------
        :class:`~nunchaku.models.utils.CPUOffloadManager`
        """
        if offload == self.offload:
            return
        self.offload = offload
        if offload:
            self.offload_manager = CPUOffloadManager(
                self.blocks,
                use_pin_memory=kwargs.get("use_pin_memory", True),
                on_gpu_modules=[
                    self.patch_embedding,
                    self.condition_embedder,
                    self.norm_out,
                    self.proj_out,
                    self.rope,
                ],
                num_blocks_on_gpu=kwargs.get("num_blocks_on_gpu", 1),
            )
        else:
            self.offload_manager = None
            gc.collect()
            torch.cuda.empty_cache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass, mirroring the diffusers Wan transformer with optional
        per-block CPU offloading.
        """
        device = hidden_states.device
        if self.offload:
            self.offload_manager.set_device(device)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        compute_stream = torch.cuda.current_stream()
        if self.offload:
            self.offload_manager.initialize(compute_stream)
        for block_idx, block in enumerate(self.blocks):
            with torch.cuda.stream(compute_stream):
                if self.offload:
                    block = self.offload_manager.get_block(block_idx)
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            if self.offload:
                self.offload_manager.step(compute_stream)

        scale_shift_table = self.scale_shift_table.to(device)
        if temb.ndim == 3:
            shift, scale = (scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if self.offload:
            torch.cuda.empty_cache()

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def to(self, *args, **kwargs):
        """
        Override the default ``.to()`` method.

        If offload is enabled, prevents moving the model to GPU.
        Prevents changing dtype after quantization.

        Raises
        ------
        ValueError
            If attempting to change dtype after quantization.
        """
        device_arg_or_kwarg_present = any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs
        dtype_present_in_args = "dtype" in kwargs

        for arg in args:
            if not isinstance(arg, str):
                continue
            try:
                torch.device(arg)
                device_arg_or_kwarg_present = True
            except RuntimeError:
                pass

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if dtype_present_in_args and self._is_initialized:
            raise ValueError(
                "Casting a quantized model to a new `dtype` is unsupported. To set the dtype of unquantized layers, "
                "please use the `torch_dtype` argument when loading the model using `from_pretrained`."
            )
        if self.offload:
            if device_arg_or_kwarg_present:
                logger.warning("Skipping moving the model to GPU as offload is enabled")
                return self
        return super(type(self), self).to(*args, **kwargs)
