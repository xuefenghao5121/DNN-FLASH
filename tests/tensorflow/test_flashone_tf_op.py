from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf

from flashone_tf import flashone_attention, select_qk_tile_layout, select_tile_sizes


def tf_reference_attention(q, k, v, *, causal: bool):
    scale = tf.cast(1.0 / math.sqrt(q.shape[-1]), q.dtype)
    scores = tf.einsum("bhmd,bhnd->bhmn", q, k) * scale
    if causal:
        m = tf.shape(q)[2]
        n = tf.shape(k)[2]
        mask = tf.linalg.band_part(tf.ones((m, n), dtype=tf.bool), -1, 0)
        scores = tf.where(mask[None, None, :, :], scores, tf.constant(-np.inf, dtype=q.dtype))
    probs = tf.nn.softmax(scores, axis=-1)
    return tf.einsum("bhmn,bhnd->bhmd", probs, v)


def test_flashone_tf_attention_matches_tensorflow_reference() -> None:
    rng = np.random.default_rng(123)
    q = tf.constant(rng.normal(size=(2, 2, 7, 5)).astype(np.float32) * 0.2)
    k = tf.constant(rng.normal(size=(2, 2, 7, 5)).astype(np.float32) * 0.2)
    v = tf.constant(rng.normal(size=(2, 2, 7, 3)).astype(np.float32) * 0.2)

    expected = tf_reference_attention(q, k, v, causal=True)
    actual = flashone_attention(q, k, v, causal=True, query_block_size=3, key_block_size=4)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


def test_select_tile_sizes_uses_sweep_heuristic() -> None:
    assert select_tile_sizes(64, 64) == (64, 64)
    assert select_tile_sizes(128, 128) == (32, 64)
    assert select_tile_sizes(256, 256) == (32, 64)
    assert select_tile_sizes(12, 9) == (12, 9)


def test_select_qk_tile_layout_uses_initial_heuristic() -> None:
    assert select_qk_tile_layout(64, 64) == "strided_k"
    assert select_qk_tile_layout(128, 128) == "strided_k"
    assert select_qk_tile_layout(256, 256) == "strided_k"


def test_flashone_tf_attention_qk_layouts_match_reference() -> None:
    rng = np.random.default_rng(321)
    q = tf.constant(rng.normal(size=(1, 2, 12, 8)).astype(np.float32) * 0.2)
    k = tf.constant(rng.normal(size=(1, 2, 12, 8)).astype(np.float32) * 0.2)
    v = tf.constant(rng.normal(size=(1, 2, 12, 4)).astype(np.float32) * 0.2)

    expected = tf_reference_attention(q, k, v, causal=True)
    for qk_tile_layout in ("strided_k", "copied_transposed"):
        actual = flashone_attention(
            q,
            k,
            v,
            causal=True,
            query_block_size=4,
            key_block_size=6,
            qk_tile_layout=qk_tile_layout,
        )
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


def test_flashone_tf_attention_brgemm_transformed_k_matches_copied() -> None:
    brgemm_op_path = Path("build-brgemm-main/flashone_tf_attention.so").resolve()
    if not brgemm_op_path.exists():
        pytest.skip("BRGEMM TensorFlow op build is not available")

    # TensorFlow does not allow loading two shared libraries that both register
    # FlashOneAttention in the same Python process. Run the BRGEMM custom-op
    # smoke in a subprocess so the default-op tests stay isolated.
    script = f"""
import numpy as np
import tensorflow as tf
from flashone_tf import flashone_attention

rng = np.random.default_rng(654)
q = tf.constant(rng.normal(size=(1, 2, 16, 16)).astype(np.float32) * 0.2)
k = tf.constant(rng.normal(size=(1, 2, 16, 16)).astype(np.float32) * 0.2)
v = tf.constant(rng.normal(size=(1, 2, 16, 8)).astype(np.float32) * 0.2)
common = dict(
    causal=True,
    query_block_size=16,
    key_block_size=16,
    tile_kernel='onednn_brgemm',
    op_path={str(brgemm_op_path)!r},
)
copied = flashone_attention(q, k, v, qk_tile_layout='copied_transposed', **common)
transformed = flashone_attention(q, k, v, qk_tile_layout='brgemm_transformed_k', **common)
np.testing.assert_allclose(transformed.numpy(), copied.numpy(), rtol=1e-5, atol=1e-5)
"""
    cwd = Path.cwd()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{cwd / 'python'}:{cwd}:{env.get('PYTHONPATH', '')}"
    subprocess.run([sys.executable, "-c", script], cwd=cwd, env=env, check=True)


def test_flashone_tf_attention_default_tile_heuristic_matches_reference() -> None:
    rng = np.random.default_rng(456)
    q = tf.constant(rng.normal(size=(1, 2, 16, 8)).astype(np.float32) * 0.2)
    k = tf.constant(rng.normal(size=(1, 2, 16, 8)).astype(np.float32) * 0.2)
    v = tf.constant(rng.normal(size=(1, 2, 16, 4)).astype(np.float32) * 0.2)

    expected = tf_reference_attention(q, k, v, causal=True)
    actual = flashone_attention(q, k, v, causal=True)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)
