from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from flashone_tf import flashone_attention, select_tile_sizes


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


def test_flashone_tf_attention_default_tile_heuristic_matches_reference() -> None:
    rng = np.random.default_rng(456)
    q = tf.constant(rng.normal(size=(1, 2, 16, 8)).astype(np.float32) * 0.2)
    k = tf.constant(rng.normal(size=(1, 2, 16, 8)).astype(np.float32) * 0.2)
    v = tf.constant(rng.normal(size=(1, 2, 16, 4)).astype(np.float32) * 0.2)

    expected = tf_reference_attention(q, k, v, causal=True)
    actual = flashone_attention(q, k, v, causal=True)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)
