from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def _default_op_path() -> Path:
    return Path(__file__).resolve().parents[2] / "build" / "flashone_tf_attention.so"


def load_flashone_op(path: str | Path | None = None):
    op_path = Path(path) if path is not None else _default_op_path()
    if not op_path.exists():
        raise FileNotFoundError(f"FlashOne TensorFlow op not found: {op_path}")
    return tf.load_op_library(str(op_path))


def flashone_attention(
    q,
    k,
    v,
    *,
    causal: bool = True,
    query_block_size: int = 16,
    key_block_size: int = 32,
    use_onednn: bool = True,
    op_path: str | Path | None = None,
):
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
