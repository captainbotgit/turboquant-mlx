"""Transparent monkey-patching of MLX-LM models for TurboQuant KV cache compression.

Usage:
    from turboquant_mlx import patch_model
    from mlx_lm import load, generate

    model, tokenizer = load("mlx-community/gemma-2b-Q4")
    model = patch_model(model, bits=3)
    print(generate(model, tokenizer, prompt="Hello", max_tokens=256))

    # Restore original behavior:
    unpatch_model(model)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn

from .qjl import QJLState
from .turbo import TurboQuantCompressor, TurboQuantConfig, TurboQuantState

logger = logging.getLogger(__name__)

# Sentinel to track patched models
_PATCHED_ATTR = "_turboquant_original_make_cache"


def _concat_states(a: TurboQuantState, b: TurboQuantState) -> TurboQuantState:
    """Concatenate two TurboQuantStates along the batch (token) dimension.

    This enables incremental compression: quantize new tokens independently
    and append to the existing compressed state without re-quantizing.
    """
    qjl_state = None
    if a.qjl_state is not None and b.qjl_state is not None:
        qjl_state = QJLState(
            sign_bits_packed=mx.concatenate(
                [a.qjl_state.sign_bits_packed, b.qjl_state.sign_bits_packed], axis=0
            ),
            sign_orig_dim=a.qjl_state.sign_orig_dim,
            norms=mx.concatenate([a.qjl_state.norms, b.qjl_state.norms], axis=0),
        )

    return TurboQuantState(
        mse_indices_packed=mx.concatenate(
            [a.mse_indices_packed, b.mse_indices_packed], axis=0
        ),
        mse_orig_dim=a.mse_orig_dim,
        norms=mx.concatenate([a.norms, b.norms], axis=0),
        qjl_state=qjl_state,
        config=a.config,
    )


class TurboQuantKVCache:
    """Drop-in replacement for mlx_lm's KVCache that compresses KV vectors.

    Stores KV cache in TurboQuant-compressed form and decompresses on-the-fly
    during attention computation. Compatible with MLX-LM's cache protocol.
    """

    step = 256  # Match KVCache allocation granularity

    def __init__(
        self,
        head_dim: int,
        n_kv_heads: int,
        config: TurboQuantConfig,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.config = config
        self.offset = 0

        # Per the paper and reference implementations:
        # - Values use MSE-only at full bit budget (they need element-wise
        #   reconstruction for weighted sum in attention)
        # - Keys use MSE-only for now (asymmetric QJL scoring, which avoids
        #   element-wise reconstruction, requires modifying the attention layer)
        # Both use mode="mse" to avoid QJL element-wise reconstruction noise.
        mse_config = TurboQuantConfig(bits=config.bits, mode="mse")
        self._k_compressors = [
            TurboQuantCompressor(head_dim, mse_config, seed=layer_idx * 1000 + h)
            for h in range(n_kv_heads)
        ]
        self._v_compressors = [
            TurboQuantCompressor(head_dim, mse_config, seed=layer_idx * 1000 + n_kv_heads + h)
            for h in range(n_kv_heads)
        ]

        # Compressed storage: list of TurboQuantState per head
        self._k_states: list[list] = [[] for _ in range(n_kv_heads)]
        self._v_states: list[list] = [[] for _ in range(n_kv_heads)]

        # Keep recent tokens uncompressed for quality (sliding window)
        self._recent_window = 32
        self._recent_keys: mx.array | None = None
        self._recent_values: mx.array | None = None

        # Full fp16 fallback for initial tokens (before compression kicks in)
        self._compress_after = 64  # Start compressing after this many tokens
        self._fp_keys: mx.array | None = None
        self._fp_values: mx.array | None = None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Store new KV pairs and return full accumulated cache for attention.

        Args:
            keys: (B, n_kv_heads, new_tokens, head_dim)
            values: (B, n_kv_heads, new_tokens, head_dim)

        Returns:
            (all_keys, all_values) for attention computation.
        """
        B, n_kv_heads, new_tokens, head_dim = keys.shape
        prev = self.offset
        self.offset += new_tokens

        # For simplicity in v0.1: keep everything in fp16 until we hit the
        # compression threshold, then compress older tokens in bulk.
        if self._fp_keys is None:
            self._fp_keys = keys
            self._fp_values = values
        else:
            self._fp_keys = mx.concatenate([self._fp_keys, keys], axis=2)
            self._fp_values = mx.concatenate([self._fp_values, values], axis=2)

        total_tokens = self._fp_keys.shape[2]

        # Compress old tokens when we exceed the threshold
        if total_tokens > self._compress_after + self._recent_window:
            n_to_compress = total_tokens - self._recent_window
            old_k = self._fp_keys[:, :, :n_to_compress, :]
            old_v = self._fp_values[:, :, :n_to_compress, :]

            # Compress per head — append new compressed to existing (no re-compression)
            for h in range(n_kv_heads):
                # (B, n_to_compress, head_dim)
                k_head = old_k[:, h, :, :]
                v_head = old_v[:, h, :, :]

                # Flatten batch and sequence for compression
                k_flat = k_head.reshape(-1, head_dim)
                v_flat = v_head.reshape(-1, head_dim)

                # Quantize only the new tokens
                new_k_state = self._k_compressors[h].quantize(k_flat)
                new_v_state = self._v_compressors[h].quantize(v_flat)

                # Append to existing compressed state (no decompress-recompress)
                if self._k_states[h]:
                    old_ks = self._k_states[h][0]
                    old_vs = self._v_states[h][0]
                    self._k_states[h] = [_concat_states(old_ks, new_k_state)]
                    self._v_states[h] = [_concat_states(old_vs, new_v_state)]
                else:
                    self._k_states[h] = [new_k_state]
                    self._v_states[h] = [new_v_state]

            # Keep only recent window in fp16
            self._fp_keys = self._fp_keys[:, :, n_to_compress:, :]
            self._fp_values = self._fp_values[:, :, n_to_compress:, :]

        # Reconstruct full KV for attention
        if any(len(s) > 0 for s in self._k_states):
            # Decompress old tokens
            decompressed_k_heads = []
            decompressed_v_heads = []
            for h in range(n_kv_heads):
                if self._k_states[h]:
                    dk = self._k_compressors[h].dequantize(self._k_states[h][0])
                    dv = self._v_compressors[h].dequantize(self._v_states[h][0])
                    # Reshape back: (B, n_compressed, head_dim)
                    n_compressed = dk.shape[0] // B
                    dk = dk.reshape(B, n_compressed, head_dim)
                    dv = dv.reshape(B, n_compressed, head_dim)
                    decompressed_k_heads.append(dk)
                    decompressed_v_heads.append(dv)
                else:
                    decompressed_k_heads.append(
                        mx.zeros((B, 0, head_dim), dtype=keys.dtype)
                    )
                    decompressed_v_heads.append(
                        mx.zeros((B, 0, head_dim), dtype=values.dtype)
                    )

            # Stack heads: (B, n_kv_heads, n_compressed, head_dim)
            old_k = mx.stack(decompressed_k_heads, axis=1)
            old_v = mx.stack(decompressed_v_heads, axis=1)

            # Concatenate with recent fp16 tokens
            all_keys = mx.concatenate([old_k, self._fp_keys], axis=2)
            all_values = mx.concatenate([old_v, self._fp_values], axis=2)
        else:
            all_keys = self._fp_keys
            all_values = self._fp_values

        return all_keys, all_values

    def size(self):
        return self.offset

    @property
    def state(self):
        # For save/load compatibility, return the fp16 representation
        if self._fp_keys is None:
            return None, None
        return self._fp_keys, self._fp_values

    @state.setter
    def state(self, v):
        self._fp_keys, self._fp_values = v
        if self._fp_keys is not None:
            self.offset = self._fp_keys.shape[2]

    @property
    def meta_state(self):
        return (str(self.offset), str(self.config.bits), str(self.config.mode))

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        # Simplified: just trim fp16 portion
        if self._fp_keys is not None and n > 0:
            trim_amount = min(n, self._fp_keys.shape[2])
            self._fp_keys = self._fp_keys[:, :, trim_amount:, :]
            self._fp_values = self._fp_values[:, :, trim_amount:, :]
        return n

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self._fp_keys is None and all(len(s) == 0 for s in self._k_states)

    @property
    def nbytes(self):
        total = 0
        if self._fp_keys is not None:
            total += self._fp_keys.nbytes + self._fp_values.nbytes
        # Compressed states are much smaller but harder to measure exactly
        # Report fp16-equivalent for now
        for h in range(self.n_kv_heads):
            for state in self._k_states[h]:
                total += state.mse_indices_packed.nbytes + state.norms.nbytes
                if state.qjl_state is not None:
                    total += state.qjl_state.sign_bits_packed.nbytes + state.qjl_state.norms.nbytes
            for state in self._v_states[h]:
                total += state.mse_indices_packed.nbytes + state.norms.nbytes
                if state.qjl_state is not None:
                    total += state.qjl_state.sign_bits_packed.nbytes + state.qjl_state.norms.nbytes
        return total


def _get_model_config(model: nn.Module) -> tuple[int, int, int]:
    """Extract head_dim, n_kv_heads, n_layers from an MLX-LM model."""
    # For multimodal models (e.g. Qwen3.5), look in language_model.args or
    # text_config first, since the top-level args may lack text model details.
    args = None
    for candidate in [
        getattr(getattr(model, "language_model", None), "args", None),
        getattr(model, "args", None),
    ]:
        if candidate is not None and hasattr(candidate, "num_hidden_layers"):
            args = candidate
            break
    if args is None and hasattr(model, "args"):
        args = model.args

    # Also check for nested text_config
    if args is not None and hasattr(args, "text_config"):
        text_cfg = args.text_config
        if hasattr(text_cfg, "head_dim") or hasattr(text_cfg, "num_hidden_layers"):
            args = text_cfg

    if args is not None:
        head_dim = getattr(args, "head_dim", None)
        if head_dim is None:
            hidden = getattr(args, "hidden_size", getattr(args, "dim", 4096))
            n_heads = getattr(args, "num_attention_heads", getattr(args, "n_heads", 32))
            head_dim = hidden // n_heads
        n_kv_heads = getattr(
            args,
            "num_key_value_heads",
            getattr(args, "n_kv_heads", getattr(args, "num_attention_heads", 32)),
        )
        n_layers = getattr(args, "num_hidden_layers", getattr(args, "n_layers", 32))
        return head_dim, n_kv_heads, n_layers

    # Fallback: inspect first attention layer
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Cannot determine model architecture — unsupported model type")

    n_layers = len(layers)
    layer0 = layers[0]
    attn = getattr(layer0, "self_attn", getattr(layer0, "attention", None))
    if attn is None:
        raise ValueError("Cannot find attention module in model layer")

    # Infer from projection weight shapes
    k_proj = getattr(attn, "k_proj", None)
    if k_proj is not None and hasattr(k_proj, "weight"):
        # weight shape: (n_kv_heads * head_dim, hidden_size)
        kv_dim = k_proj.weight.shape[0]
        q_proj = attn.q_proj
        q_dim = q_proj.weight.shape[0]
        hidden = q_proj.weight.shape[1]
        n_heads = getattr(model.args, "num_attention_heads", q_dim // (kv_dim // 1))
        # Try to determine head_dim
        if hasattr(attn, "n_kv_heads"):
            n_kv_heads = attn.n_kv_heads
            head_dim = kv_dim // n_kv_heads
        else:
            head_dim = 128  # common default
            n_kv_heads = kv_dim // head_dim

    return head_dim, n_kv_heads, n_layers


def patch_model(
    model: nn.Module,
    bits: int = 3,
    mode: Literal["prod", "mse"] = "prod",
) -> nn.Module:
    """Patch an MLX-LM model to use TurboQuant KV cache compression.

    This replaces the model's cache construction so that all subsequent
    calls to generate() use compressed KV caches automatically.

    Args:
        model: An MLX-LM model (e.g., from mlx_lm.load()).
        bits: Quantization bits per dimension (default 3).
        mode: "prod" for full TurboQuant (MSE + QJL), "mse" for MSE-only.

    Returns:
        The same model object, patched in-place.
    """
    if hasattr(model, _PATCHED_ATTR):
        logger.warning("Model already patched — unpatching first")
        unpatch_model(model)

    head_dim, n_kv_heads, n_layers = _get_model_config(model)
    config = TurboQuantConfig(bits=bits, mode=mode)

    logger.info(
        f"Patching model: {n_layers} layers, {n_kv_heads} KV heads, "
        f"head_dim={head_dim}, bits={bits}, mode={mode}"
    )

    # Save original make_cache if it exists
    inner_model = model.model if hasattr(model, "model") else model
    original_make_cache = getattr(inner_model, "make_cache", None)

    # Detect SSM (non-attention) layer indices that need standard caches.
    # Models like Qwen3.5 have hybrid attention+SSM architectures.
    _inner_for_ssm = getattr(model, "language_model", inner_model)
    _inner_for_ssm = getattr(_inner_for_ssm, "model", _inner_for_ssm)
    ssm_indices = set()
    if hasattr(_inner_for_ssm, "ssm_idx"):
        idx = _inner_for_ssm.ssm_idx
        if isinstance(idx, (list, tuple)):
            ssm_indices = set(idx)
        elif isinstance(idx, int):
            ssm_indices = {idx}
    if ssm_indices:
        logger.info(f"SSM layer indices (will use standard cache): {sorted(ssm_indices)}")

    def turboquant_make_cache():
        # If the model has its own make_cache, use it as the base and only
        # replace attention (KV) caches with TurboQuantKVCache.
        if original_make_cache is not None:
            base_caches = original_make_cache()
            from mlx_lm.models.cache import KVCache

            caches = []
            for i, c in enumerate(base_caches):
                if isinstance(c, KVCache):
                    caches.append(TurboQuantKVCache(head_dim, n_kv_heads, config, layer_idx=i))
                else:
                    caches.append(c)  # Keep SSM/ArraysCache etc. as-is
            return caches

        return [
            TurboQuantKVCache(head_dim, n_kv_heads, config, layer_idx=i)
            for i in range(n_layers)
        ]

    # Patch make_cache on both the top-level model and inner model.
    # make_prompt_cache checks hasattr(model, "make_cache") on the top-level object.
    setattr(model, "make_cache", turboquant_make_cache)
    if hasattr(model, "model"):
        setattr(model.model, "make_cache", turboquant_make_cache)

    # Store original for unpatch
    setattr(model, _PATCHED_ATTR, original_make_cache)
    setattr(model, "_turboquant_config", config)

    return model


def unpatch_model(model: nn.Module) -> nn.Module:
    """Remove TurboQuant patching and restore original cache behavior.

    Args:
        model: A previously patched MLX-LM model.

    Returns:
        The same model object, restored.
    """
    if not hasattr(model, _PATCHED_ATTR):
        logger.warning("Model is not patched — nothing to do")
        return model

    original = getattr(model, _PATCHED_ATTR)
    inner_model = model.model if hasattr(model, "model") else model

    if original is not None:
        inner_model.make_cache = original
        if hasattr(model, "make_cache") and model is not inner_model:
            model.make_cache = original
    else:
        if hasattr(inner_model, "make_cache"):
            delattr(inner_model, "make_cache")
        if model is not inner_model and hasattr(model, "make_cache"):
            delattr(model, "make_cache")

    delattr(model, _PATCHED_ATTR)
    if hasattr(model, "_turboquant_config"):
        delattr(model, "_turboquant_config")

    logger.info("Model unpatched — original cache behavior restored")
    return model
