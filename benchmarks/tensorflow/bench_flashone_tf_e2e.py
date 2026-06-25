#!/usr/bin/env python3
"""TensorFlow E2E benchmark for the FlashOne custom op MVP.

This intentionally benchmarks the current CPU custom-op path, not XLA lowering.
It compares:
  1. TensorFlow materialized scaled-dot-product attention
  2. FlashOne custom op attention
  3. A tiny decoder block using each attention implementation
"""
from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from flashone_tf import flashone_attention


@dataclass(frozen=True)
class BenchConfig:
    batch: int
    heads: int
    seq: int
    head_dim: int
    ffn_multiplier: int
    causal: bool
    query_block: int
    key_block: int
    warmup: int
    repeat: int

    @property
    def embed_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def ffn_dim(self) -> int:
        return self.embed_dim * self.ffn_multiplier


def stable_values(shape: tuple[int, ...], *, seed: int, scale: float = 0.05) -> tf.Tensor:
    rng = np.random.default_rng(seed)
    return tf.constant(rng.normal(size=shape).astype(np.float32) * scale)


def tf_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, *, causal: bool) -> tf.Tensor:
    scale = tf.cast(1.0 / math.sqrt(int(q.shape[-1])), q.dtype)
    scores = tf.matmul(q, k, transpose_b=True) * scale
    if causal:
        m = tf.shape(q)[2]
        n = tf.shape(k)[2]
        mask = tf.linalg.band_part(tf.ones((m, n), dtype=tf.bool), -1, 0)
        scores = tf.where(mask[None, None, :, :], scores, tf.constant(-np.inf, dtype=q.dtype))
    probs = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(probs, v)


def flashone_attention_tf(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, cfg: BenchConfig) -> tf.Tensor:
    return flashone_attention(
        q,
        k,
        v,
        causal=cfg.causal,
        query_block_size=cfg.query_block,
        key_block_size=cfg.key_block,
        use_onednn=True,
    )


def split_qkv(qkv: tf.Tensor, cfg: BenchConfig) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    q, k, v = tf.split(qkv, 3, axis=-1)

    def to_heads(x: tf.Tensor) -> tf.Tensor:
        x = tf.reshape(x, (cfg.batch, cfg.seq, cfg.heads, cfg.head_dim))
        return tf.transpose(x, (0, 2, 1, 3))

    return to_heads(q), to_heads(k), to_heads(v)


def merge_heads(x: tf.Tensor, cfg: BenchConfig) -> tf.Tensor:
    x = tf.transpose(x, (0, 2, 1, 3))
    return tf.reshape(x, (cfg.batch, cfg.seq, cfg.embed_dim))


def layer_norm(x: tf.Tensor, gamma: tf.Tensor, beta: tf.Tensor, eps: float = 1e-5) -> tf.Tensor:
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * tf.math.rsqrt(var + eps) * gamma + beta


def make_decoder_weights(cfg: BenchConfig) -> dict[str, tf.Tensor]:
    e = cfg.embed_dim
    f = cfg.ffn_dim
    return {
        "w_qkv": stable_values((e, 3 * e), seed=1),
        "b_qkv": stable_values((3 * e,), seed=2),
        "w_o": stable_values((e, e), seed=3),
        "b_o": stable_values((e,), seed=4),
        "ln1_gamma": tf.ones((e,), dtype=tf.float32),
        "ln1_beta": tf.zeros((e,), dtype=tf.float32),
        "w_ff1": stable_values((e, f), seed=5),
        "b_ff1": stable_values((f,), seed=6),
        "w_ff2": stable_values((f, e), seed=7),
        "b_ff2": stable_values((e,), seed=8),
        "ln2_gamma": tf.ones((e,), dtype=tf.float32),
        "ln2_beta": tf.zeros((e,), dtype=tf.float32),
    }


def decoder_block(
    x: tf.Tensor,
    weights: dict[str, tf.Tensor],
    cfg: BenchConfig,
    attention_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
) -> tf.Tensor:
    qkv = tf.matmul(x, weights["w_qkv"]) + weights["b_qkv"]
    q, k, v = split_qkv(qkv, cfg)
    attn = attention_fn(q, k, v)
    attn = merge_heads(attn, cfg)
    x = x + tf.matmul(attn, weights["w_o"]) + weights["b_o"]
    x = layer_norm(x, weights["ln1_gamma"], weights["ln1_beta"])
    ffn = tf.nn.gelu(tf.matmul(x, weights["w_ff1"]) + weights["b_ff1"])
    x = x + tf.matmul(ffn, weights["w_ff2"]) + weights["b_ff2"]
    return layer_norm(x, weights["ln2_gamma"], weights["ln2_beta"])


def benchmark(name: str, fn: Callable[[], tf.Tensor], *, warmup: int, repeat: int) -> tuple[float, tf.Tensor]:
    out = fn()
    # Force op-library load and initial Eigen/oneDNN setup outside the timing loop.
    _ = out.numpy()
    for _ in range(warmup):
        _ = fn().numpy()
    start = time.perf_counter()
    for _ in range(repeat):
        out = fn()
        _ = out.numpy()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeat
    print(f"{name}_ms: {elapsed_ms:.6f}")
    return elapsed_ms, out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--query-block", type=int, default=16)
    parser.add_argument("--key-block", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--graph", action="store_true", help="Wrap benchmark functions with tf.function(jit_compile=False)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchConfig(
        batch=args.batch,
        heads=args.heads,
        seq=args.seq,
        head_dim=args.head_dim,
        ffn_multiplier=args.ffn_multiplier,
        causal=not args.no_causal,
        query_block=args.query_block,
        key_block=args.key_block,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print("FlashOne TensorFlow E2E benchmark")
    print(f"mode: {'graph' if args.graph else 'eager'}")
    print(
        f"shape: B={cfg.batch} H={cfg.heads} M=N={cfg.seq} D={cfg.head_dim} "
        f"E={cfg.embed_dim} causal={cfg.causal} q_block={cfg.query_block} k_block={cfg.key_block}"
    )
    print(f"tensorflow_version: {tf.__version__}")

    q = stable_values((cfg.batch, cfg.heads, cfg.seq, cfg.head_dim), seed=11)
    k = stable_values((cfg.batch, cfg.heads, cfg.seq, cfg.head_dim), seed=12)
    v = stable_values((cfg.batch, cfg.heads, cfg.seq, cfg.head_dim), seed=13)
    x = stable_values((cfg.batch, cfg.seq, cfg.embed_dim), seed=14)
    weights = make_decoder_weights(cfg)

    tf_attn_out = tf_attention(q, k, v, causal=cfg.causal)
    flash_attn_out = flashone_attention_tf(q, k, v, cfg)
    attn_max_abs_diff = tf.reduce_max(tf.abs(tf_attn_out - flash_attn_out)).numpy()
    print(f"attention_max_abs_diff: {attn_max_abs_diff:.9g}")

    tf_decoder_out = decoder_block(
        x,
        weights,
        cfg,
        lambda q_, k_, v_: tf_attention(q_, k_, v_, causal=cfg.causal),
    )
    flash_decoder_out = decoder_block(
        x,
        weights,
        cfg,
        lambda q_, k_, v_: flashone_attention_tf(q_, k_, v_, cfg),
    )
    decoder_max_abs_diff = tf.reduce_max(tf.abs(tf_decoder_out - flash_decoder_out)).numpy()
    print(f"decoder_max_abs_diff: {decoder_max_abs_diff:.9g}")

    tf_attention_fn = lambda: tf_attention(q, k, v, causal=cfg.causal)
    flashone_attention_fn = lambda: flashone_attention_tf(q, k, v, cfg)
    tf_decoder_fn = lambda: decoder_block(
        x,
        weights,
        cfg,
        lambda q_, k_, v_: tf_attention(q_, k_, v_, causal=cfg.causal),
    )
    flashone_decoder_fn = lambda: decoder_block(
        x,
        weights,
        cfg,
        lambda q_, k_, v_: flashone_attention_tf(q_, k_, v_, cfg),
    )
    if args.graph:
        tf_attention_fn = tf.function(tf_attention_fn, jit_compile=False, autograph=False)
        flashone_attention_fn = tf.function(flashone_attention_fn, jit_compile=False, autograph=False)
        tf_decoder_fn = tf.function(tf_decoder_fn, jit_compile=False, autograph=False)
        flashone_decoder_fn = tf.function(flashone_decoder_fn, jit_compile=False, autograph=False)

    benchmark("tensorflow_attention", tf_attention_fn, warmup=cfg.warmup, repeat=cfg.repeat)
    benchmark("flashone_attention", flashone_attention_fn, warmup=cfg.warmup, repeat=cfg.repeat)
    benchmark("tensorflow_decoder_block", tf_decoder_fn, warmup=cfg.warmup, repeat=cfg.repeat)
    benchmark("flashone_decoder_block", flashone_decoder_fn, warmup=cfg.warmup, repeat=cfg.repeat)


if __name__ == "__main__":
    main()
