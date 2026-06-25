from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

ScoreBias = Callable[[int, int], float]
BlockMask = np.ndarray


def _validate(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q, k, and v must be 2D arrays")
    if q.shape[1] != k.shape[1]:
        raise ValueError("q and k head dimensions must match")
    if k.shape[0] != v.shape[0]:
        raise ValueError("k and v token dimensions must match")
    return q, k, v


def _apply_score_mods(
    scores: np.ndarray,
    qi: int,
    *,
    causal: bool,
    block_mask: BlockMask | None,
    query_block_size: int,
    key_block_size: int,
    score_bias: ScoreBias | None,
    key_offset: int = 0,
) -> np.ndarray:
    scores = scores.astype(np.float32, copy=True)
    key_indices = np.arange(key_offset, key_offset + scores.shape[0])
    if causal:
        scores = np.where(key_indices > qi, -np.inf, scores)
    if block_mask is not None:
        if query_block_size <= 0 or key_block_size <= 0:
            raise ValueError("block sizes must be positive")
        qb = qi // query_block_size
        kb = key_indices // key_block_size
        allowed = np.array(
            [qb < block_mask.shape[0] and col < block_mask.shape[1] and block_mask[qb, col] for col in kb],
            dtype=bool,
        )
        scores = np.where(allowed, scores, -np.inf)
    if score_bias is not None:
        bias = np.array([score_bias(qi, int(kj)) for kj in key_indices], dtype=np.float32)
        scores = np.where(np.isneginf(scores), scores, scores + bias)
    return scores


def standard_attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    scale: float | None = None,
    causal: bool = False,
    block_mask: BlockMask | None = None,
    query_block_size: int = 1,
    key_block_size: int = 1,
    score_bias: ScoreBias | None = None,
) -> np.ndarray:
    """Reference scaled dot-product attention that materializes the score matrix."""

    q, k, v = _validate(q, k, v)
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[1])
    scores = q @ k.T * np.float32(scale)
    for qi in range(scores.shape[0]):
        scores[qi] = _apply_score_mods(
            scores[qi],
            qi,
            causal=causal,
            block_mask=block_mask,
            query_block_size=query_block_size,
            key_block_size=key_block_size,
            score_bias=score_bias,
        )
    row_max = np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores - row_max)
    probs = np.where(np.isneginf(scores), np.float32(0.0), probs)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return (probs @ v).astype(np.float32)


def flash_attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    scale: float | None = None,
    causal: bool = False,
    key_block_size: int = 64,
    block_mask: BlockMask | None = None,
    query_block_size: int = 1,
    block_mask_key_block_size: int | None = None,
    score_bias: ScoreBias | None = None,
) -> np.ndarray:
    """Flash-style tiled attention using online softmax recurrence."""

    q, k, v = _validate(q, k, v)
    if key_block_size <= 0:
        raise ValueError("key_block_size must be positive")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[1])
    if block_mask_key_block_size is None:
        block_mask_key_block_size = key_block_size

    query_tokens, head_dim = q.shape
    key_tokens = k.shape[0]
    value_dim = v.shape[1]
    out = np.zeros((query_tokens, value_dim), dtype=np.float32)

    for qi in range(query_tokens):
        running_max = -np.inf
        running_sum = np.float32(0.0)
        acc = np.zeros((value_dim,), dtype=np.float32)

        for block_begin in range(0, key_tokens, key_block_size):
            if causal and block_begin > qi:
                break
            block_end = min(key_tokens, block_begin + key_block_size)
            block_k = k[block_begin:block_end]
            scores = (block_k @ q[qi].reshape(head_dim, 1)).reshape(-1) * np.float32(scale)
            scores = _apply_score_mods(
                scores,
                qi,
                causal=causal,
                block_mask=block_mask,
                query_block_size=query_block_size,
                key_block_size=block_mask_key_block_size,
                score_bias=score_bias,
                key_offset=block_begin,
            )
            if np.all(np.isneginf(scores)):
                continue
            block_max = np.max(scores)
            new_max = max(running_max, block_max)
            old_scale = np.float32(0.0) if np.isneginf(running_max) else np.exp(running_max - new_max)
            weights = np.exp(scores - new_max).astype(np.float32)
            weights = np.where(np.isneginf(scores), np.float32(0.0), weights)

            acc *= old_scale
            running_sum *= old_scale
            running_sum += np.sum(weights, dtype=np.float32)
            acc += weights @ v[block_begin:block_end]
            running_max = new_max

        out[qi] = acc / running_sum

    return out.astype(np.float32)
