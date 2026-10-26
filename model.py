"""Flax/Linen pixel-space DiT. Images use NHWC; parameters stay float32.

Architecture adapted from https://github.com/facebookresearch/DiT.
"""

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


def timestep_embedding(t, dim=256, freq_scale=1.0):
    half = dim // 2
    freqs = jnp.exp(-math.log(10000) * jnp.arange(half, dtype=jnp.float32) / half)
    args = t.astype(jnp.float32)[:, None] * freqs[None] * freq_scale
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return jnp.pad(embedding, ((0, 0), (0, dim % 2)))


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    if embed_dim % 4:
        raise ValueError("hidden_size must be divisible by 4")
    grid = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    omega = 1.0 / 10000 ** (np.arange(embed_dim // 4, dtype=np.float64) / (embed_dim / 4))
    embeddings = []
    for pos in grid:
        angles = pos.reshape(-1, 1) * omega[None]
        embeddings.extend([np.sin(angles), np.cos(angles)])
    return np.concatenate(embeddings, axis=-1).astype(np.float32)


def dense(features, dtype, name, zero=False, normal=False):
    init = nn.initializers.zeros if zero else (
        nn.initializers.normal(0.02) if normal else nn.initializers.xavier_uniform()
    )
    return nn.Dense(features, dtype=dtype, param_dtype=jnp.float32, kernel_init=init, name=name)


class Attention(nn.Module):
    hidden_size: int
    num_heads: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        b, n, d = x.shape
        qkv = dense(3 * d, self.dtype, "qkv")(x).reshape(b, n, 3, self.num_heads, d // self.num_heads)
        q, k, v = (qkv[:, :, i] for i in range(3))
        # Portable XLA attention; softmax accumulates in float32 in JAX.
        x = jax.nn.dot_product_attention(q, k, v, implementation="xla")
        return dense(d, self.dtype, "proj")(x.reshape(b, n, d))


class DiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = jnp.split(
            dense(6 * self.hidden_size, self.dtype, "adaLN", zero=True)(nn.silu(c)), 6, axis=-1
        )
        norm = lambda name: nn.LayerNorm(
            use_bias=False, use_scale=False, epsilon=1e-6, dtype=self.dtype,
            force_float32_reductions=True, name=name,
        )
        h = modulate(norm("norm1")(x), shift_a, scale_a)
        x = x + gate_a[:, None] * Attention(self.hidden_size, self.num_heads, self.dtype, name="attn")(h)
        h = modulate(norm("norm2")(x), shift_m, scale_m)
        h = dense(int(self.hidden_size * self.mlp_ratio), self.dtype, "fc1")(h)
        h = nn.gelu(h, approximate=False)
        h = dense(self.hidden_size, self.dtype, "fc2")(h)
        return x + gate_m[:, None] * h


class NanoDiT(nn.Module):
    input_size: int = 64
    patch_size: int = 2
    in_channels: int = 3
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 75
    timestep_freq_scale: float = 1000.0
    class_dropout_prob: float = 0.1
    dtype: Any = jnp.bfloat16
    remat: bool = True

    @nn.compact
    def __call__(self, x, t, y, *, train=False):
        if self.input_size % self.patch_size or self.hidden_size % self.num_heads:
            raise ValueError("image size must divide into patches; hidden size must divide into heads")
        if x.shape[1:] != (self.input_size, self.input_size, self.in_channels):
            raise ValueError("expected NHWC images matching the model configuration")
        b, p = x.shape[0], self.patch_size
        g = self.input_size // p
        # Flattened patches + Dense is a strided convolution with linear Xavier init.
        x = x.reshape(b, g, p, g, p, self.in_channels).transpose(0, 1, 3, 2, 4, 5)
        x = dense(self.hidden_size, self.dtype, "patch_embed")(x.reshape(b, g * g, -1))
        pos = get_2d_sincos_pos_embed(self.hidden_size, g)
        x = x + jnp.asarray(pos, dtype=self.dtype)[None]
        t = timestep_embedding(t, freq_scale=self.timestep_freq_scale)
        t = dense(self.hidden_size, self.dtype, "time_fc1", normal=True)(t)
        t = dense(self.hidden_size, self.dtype, "time_fc2", normal=True)(nn.silu(t))
        if train and self.class_dropout_prob > 0:
            drop = jax.random.bernoulli(self.make_rng("dropout"), self.class_dropout_prob, y.shape)
            y = jnp.where(drop, self.num_classes, y)
        # Always reserve the null class, including when dropout is disabled.
        y = nn.Embed(self.num_classes + 1, self.hidden_size, dtype=self.dtype,
                     embedding_init=nn.initializers.normal(0.02), name="label_embed")(y)
        c = t + y
        block = nn.remat(DiTBlock) if self.remat else DiTBlock
        for i in range(self.depth):
            x = block(self.hidden_size, self.num_heads, self.mlp_ratio, self.dtype, name=f"blocks_{i}")(x, c)
        shift, scale = jnp.split(dense(2 * self.hidden_size, self.dtype, "final_adaLN", zero=True)(nn.silu(c)), 2, -1)
        x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6,
                         dtype=self.dtype, force_float32_reductions=True, name="final_norm")(x)
        x = dense(p * p * self.in_channels, self.dtype, "final_linear", zero=True)(modulate(x, shift, scale))
        x = x.reshape(b, g, g, p, p, self.in_channels).transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(b, self.input_size, self.input_size, self.in_channels).astype(jnp.float32)
