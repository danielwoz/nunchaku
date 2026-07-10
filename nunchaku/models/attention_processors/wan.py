"""
Attention processors for :class:`~nunchaku.models.transformers.transformer_wan.NunchakuWanAttention`.
"""

from typing import Optional, Tuple

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn


def _apply_rotary_emb(
    hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    # interleaved-pair rotary embedding, matching diffusers' WanAttnProcessor
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class NunchakuWanNaiveFA2Processor:
    """
    Naive attention processor for Wan 2.1 self- and cross-attention.

    Computes quantized QKV projections, applies across-heads QK normalization
    and 3D rotary embeddings (self-attention only), dispatches to the attention
    backend, and applies the quantized output projection.
    """

    _attention_backend = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for Wan attention.

        Parameters
        ----------
        attn : :class:`~nunchaku.models.transformers.transformer_wan.NunchakuWanAttention`
            Attention module.
        hidden_states : torch.Tensor, shape (B, L, H*D)
            Video token stream.
        encoder_hidden_states : torch.Tensor, optional
            Text token stream (cross-attention only). For I2V checkpoints the
            leading tokens are the CLIP image stream.
        attention_mask : torch.Tensor, optional
            Attention mask.
        rotary_emb : tuple of torch.Tensor, optional
            (freqs_cos, freqs_sin) rotary embeddings (self-attention only).

        Returns
        -------
        torch.Tensor, shape (B, L, H*D)
            Attention output after the quantized output projection.
        """
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded as in diffusers
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        if attn.is_cross_attention:
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        else:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

        # across-heads RMSNorm applies on the flat (B, L, H*D) layout
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        # I2V image stream (unquantized bf16 projections over ~257 CLIP tokens)
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states_img))
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
