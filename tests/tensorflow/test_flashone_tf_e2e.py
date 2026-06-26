from __future__ import annotations

import numpy as np
import tensorflow as tf

from benchmarks.tensorflow.bench_flashone_tf_e2e import (
    BenchConfig,
    decoder_block,
    flashone_attention_tf,
    make_decoder_weights,
    stable_values,
    tf_attention,
)


def test_flashone_custom_op_can_replace_attention_in_decoder_block() -> None:
    cfg = BenchConfig(
        batch=1,
        heads=2,
        seq=8,
        head_dim=4,
        ffn_multiplier=2,
        causal=True,
        query_block=4,
        key_block=4,
        qk_tile_layout="copied_transposed",
        warmup=1,
        repeat=1,
    )
    x = stable_values((cfg.batch, cfg.seq, cfg.embed_dim), seed=101)
    weights = make_decoder_weights(cfg)

    expected = decoder_block(
        x,
        weights,
        cfg,
        lambda q, k, v: tf_attention(q, k, v, causal=cfg.causal),
    )
    actual = decoder_block(
        x,
        weights,
        cfg,
        lambda q, k, v: flashone_attention_tf(q, k, v, cfg),
    )

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-5, atol=2e-5)
