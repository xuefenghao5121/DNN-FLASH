from __future__ import annotations

import math

import numpy as np

from flashone import flash_attention_reference, standard_attention_reference


def test_block_mask_and_score_bias_match_standard() -> None:
    rng = np.random.default_rng(42)
    q = rng.normal(size=(6, 4)).astype(np.float32) * 0.3
    k = rng.normal(size=(8, 4)).astype(np.float32) * 0.3
    v = rng.normal(size=(8, 3)).astype(np.float32) * 0.3
    block_mask = np.array(
        [
            [True, False, False, False],
            [True, True, False, False],
            [False, True, True, True],
        ],
        dtype=bool,
    )

    def bias(qi: int, kj: int) -> float:
        return qi * 0.03 - kj * 0.01

    expected = standard_attention_reference(
        q,
        k,
        v,
        scale=1.0 / math.sqrt(q.shape[1]),
        block_mask=block_mask,
        query_block_size=2,
        key_block_size=2,
        score_bias=bias,
    )
    actual = flash_attention_reference(
        q,
        k,
        v,
        scale=1.0 / math.sqrt(q.shape[1]),
        key_block_size=3,
        block_mask=block_mask,
        query_block_size=2,
        block_mask_key_block_size=2,
        score_bias=bias,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
