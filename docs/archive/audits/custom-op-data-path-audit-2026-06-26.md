# OneDNN-Flash Custom Op Data Path Audit — 2026-06-26

## Scope

Audited the current TensorFlow CPU Custom Op path:

`tensorflow_ops/onednn_flash_attention_op.cc`
→ `flash_attention_batched_qk_pv_tile(...)`
→ `flash_attention_qk_pv_tile_ws(...)`
→ `matmul_tile_*` / oneDNN tile kernels.

## Current data path

### TensorFlow op boundary

- Inputs are read from TensorFlow tensors via `q.flat<float>().data()`, `k.flat<float>().data()`, `v.flat<float>().data()`.
- Output tensor is allocated once by TensorFlow via `ctx->allocate_output(...)`.
- OneDNN-Flash writes directly to `out->flat<float>().data()`.
- No TensorFlow-side Q/K/V copies were found.

### Batched/head wrapper

- `flash_attention_batched_qk_pv_tile(...)` flattens `[B,H,...]` by pointer offsets.
- It uses one `thread_local AttentionWorkspace` and reuses it across all `(batch, head)` pairs in the invoking thread.
- Per `(B,H)` call passes raw Q/K/V/O pointers to the inner kernel.

### Workspace kernel

- Q tile is zero-copy: `q_tile_ptr = q + q_block_begin * head_dim`.
- V tile is zero-copy: `v_tile_ptr = v + k_block_begin * value_dim`.
- O is direct-write: normalized output writes directly to `output[qi * value_dim + vd]`.
- Before this patch, K tile was explicitly transposed into `ws.k_tile_t` for every K block.

### oneDNN tile kernel

- Primitive descriptors are cached by shape.
- Before this patch, cache lookup and primitive execution were both under the same global mutex.
- oneDNN `dnnl::memory` wrappers are still constructed per tile call.

## Patch: strided K view for QK matmul

Implemented `StridedMatmulShape` and `matmul_tile_strided_inplace(...)`.

The QK tile path now uses a strided oneDNN B memory descriptor over the row-major K tensor:

- A: Q view `[q_rows, head_dim]`, contiguous row-major.
- B: K view interpreted as `[head_dim, k_cols]` with strides `{1, head_dim}`.
- C: score tile `[q_rows, k_cols]`, contiguous row-major.

This removes the inner-loop explicit K transpose/copy.

## Patch: narrower oneDNN cache lock

The oneDNN tile cache mutex now protects only cache lookup/insertion. Primitive execution runs outside the cache mutex.

Note: the current implementation still uses a shared oneDNN stream, so future true parallel execution may need per-thread streams or a stronger stream/thread-safety design.

## Verification

```text
cmake --build build -j
PYTHONPATH=python:. python3 -m pytest -q tests/python tests/tensorflow
14 passed

ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 7
```

C++ microbenchmark after strided K:

```text
max_abs_diff_qk_pv_onednn=1.86265e-08
flash_attention_qk_pv_onednn_ms=0.772644
```

TensorFlow eager benchmark after strided K, heuristic tiles:

| Seq | Tile | TF attention (ms) | OneDNN-Flash attention (ms) | Notes |
|---:|---|---:|---:|---|
| 128 | 32x64 | 2.422252 | 1.798738 | Still faster than TF, slightly slower than best pre-patch noisy sweep `1.645492ms` |
| 256 | 32x64 | 1.490950 | 4.807103 | Better than pre-patch sweep best `5.207241ms` |

## Interpretation

- The data path is already mostly zero-copy at TensorFlow and B/H wrapper boundaries.
- Removing the explicit K transpose is structurally cleaner and helps seq256, but oneDNN strided-B performance is not uniformly better for smaller seq128.
- The next likely bottlenecks are:
  1. oneDNN memory wrapper construction per tile,
  2. shared stream / stream wait per tile,
  3. scalar online softmax loops,
  4. TensorFlow Custom Op call overhead in graph mode.

## Next optimization candidates

1. Cache memory wrappers or introduce a tile execution context that owns reusable `dnnl::memory` objects per shape.
2. Evaluate copying K transpose vs strided-B as a runtime/heuristic choice. For some shapes, contiguous B may still beat strided B despite the copy.
3. Batch more work per oneDNN call or move toward a BRGEMM-style primitive to reduce per-tile `execute + wait` overhead.
4. Start XLA Custom Call minimal path to address graph-mode overhead.

## Follow-up: copied-K vs strided-K dual path

A follow-up patch retained both QK layouts:

- `strided_k`: no K transpose/copy; oneDNN B memory uses strides `{1, head_dim}`.
- `copied_transposed`: materializes contiguous K^T into workspace, then calls the regular contiguous tile matmul.

The layout is now exposed as:

- C++: `AttentionOptions::qk_tile_layout`
- TensorFlow op attr: `qk_tile_layout`
- Python wrapper arg: `qk_tile_layout`
- Benchmark CLI: `--qk-tile-layout` and `--qk-layouts`

Snapshot results:

| Shape | Layout | OneDNN-Flash attention (ms) |
|---|---|---:|
| C++ M=N=128,D=64,V=64 | strided_k | 0.750128 |
| C++ M=N=128,D=64,V=64 | copied_transposed | 0.742123 |
| TF eager seq64,D32 | strided_k | 1.512712 |
| TF eager seq64,D32 | copied_transposed | 2.024155 |
| TF eager seq128,D32 | strided_k | 1.972772 |
| TF eager seq128,D32 | copied_transposed | 2.441254 |
| TF eager seq256,D32 | strided_k | 4.727749 |
| TF eager seq256,D32 | copied_transposed | 5.103270 |
| TF eager seq128,D64 | strided_k | 2.201044 |
| TF eager seq128,D64 | copied_transposed | 3.678657 |

Decision: keep `copied_transposed` as an explicit experimental path, but default the TensorFlow wrapper heuristic to `strided_k` because it wins in the current custom-op benchmark path.
