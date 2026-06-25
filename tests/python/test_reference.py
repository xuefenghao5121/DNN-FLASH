from __future__ import annotations

import math

import numpy as np
import pytest

from flashone import flash_attention_reference, standard_attention_reference


def make_inputs(m: int, n: int, kdim: int, vdim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260625)
    q = rng.normal(0.0, 0.4, size=(m, kdim)).astype(np.float32)
    k = rng.normal(0.0, 0.4, size=(n, kdim)).astype(np.float32)
    v = rng.normal(0.0, 0.4, size=(n, vdim)).astype(np.float32)
    return q, k, v


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("block", [1, 2, 3, 8])
def test_flash_reference_matches_standard(causal: bool, block: int) -> None:
    q, k, v = make_inputs(7, 9, 5, 4)
    scale = 1.0 / math.sqrt(q.shape[1])
    expected = standard_attention_reference(q, k, v, scale=scale, causal=causal)
    actual = flash_attention_reference(q, k, v, scale=scale, causal=causal, key_block_size=block)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_invalid_block_size() -> None:
    q, k, v = make_inputs(2, 2, 2, 2)
    with pytest.raises(ValueError, match="key_block_size"):
        flash_attention_reference(q, k, v, key_block_size=0)
