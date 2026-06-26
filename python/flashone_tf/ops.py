from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import tensorflow as tf


def _default_op_path() -> Path:
    return Path(__file__).resolve().parents[2] / "build" / "flashone_tf_attention.so"


def select_tile_sizes(query_tokens: int, key_tokens: int | None = None) -> tuple[int, int]:
    """Select default FlashOne tile sizes for the current TensorFlow custom-op path.

    The heuristic is intentionally small and conservative. It comes from the
    2026-06-26 CPU eager sweep at B=1,H=4,D=32,E=128,causal=true:
      - seq64 best observed at 64x64
      - seq128/seq256 best observed around 32x64

    Callers may still pass explicit query/key block sizes to override it.
    """
    if query_tokens <= 0:
        raise ValueError("query_tokens must be positive")
    key_tokens = query_tokens if key_tokens is None else key_tokens
    if key_tokens <= 0:
        raise ValueError("key_tokens must be positive")

    if query_tokens <= 64 and key_tokens <= 64:
        return min(64, query_tokens), min(64, key_tokens)
    return min(32, query_tokens), min(64, key_tokens)


@lru_cache(maxsize=None)
def _load_flashone_op_cached(op_path: str):
    path = Path(op_path)
    if not path.exists():
        raise FileNotFoundError(f"FlashOne TensorFlow op not found: {path}")
    return tf.load_op_library(str(path))


def load_flashone_op(path: str | Path | None = None):
    op_path = Path(path) if path is not None else _default_op_path()
    return _load_flashone_op_cached(str(op_path.resolve()))


def flashone_attention(
    q,
    k,
    v,
    *,
    causal: bool = True,
    query_block_size: int | None = None,
    key_block_size: int | None = None,
    use_onednn: bool = True,
    op_path: str | Path | None = None,
):
    if query_block_size is None or key_block_size is None:
        q_shape = tf.TensorShape(q.shape)
        k_shape = tf.TensorShape(k.shape)
        if q_shape.rank != 4 or k_shape.rank != 4:
            raise ValueError("q and k must be rank 4 [B,H,T,D] tensors")
        if q_shape[2] is None or k_shape[2] is None:
            raise ValueError(
                "query_block_size/key_block_size must be provided when q/k token dimensions are dynamic"
            )
        heuristic_q, heuristic_k = select_tile_sizes(int(q_shape[2]), int(k_shape[2]))
        query_block_size = heuristic_q if query_block_size is None else query_block_size
        key_block_size = heuristic_k if key_block_size is None else key_block_size

    ops = load_flashone_op(op_path)
    return ops.flash_one_attention(
        q,
        k,
        v,
        causal=causal,
        query_block_size=query_block_size,
        key_block_size=key_block_size,
        use_onednn=use_onednn,
    )
