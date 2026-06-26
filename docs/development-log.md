# FlashOne Development Log

## 2026-06-25 — Project start

### Done

- Renamed generic scaffold to `FlashOne`.
- Imported the full project plan into `docs/project-plan.md`.
- Added requirements, architecture design, and ADR-0001.
- Implemented standalone C++17 attention core:
  - `standard_attention`
  - `flash_attention_tiled`
  - causal mask
  - online softmax recurrence
- Implemented Python/Numpy golden reference.
- Added C++ and Python correctness tests.
- Added minimal benchmark executable.

### Verification

```text
PYTHONPATH=python python3 -m pytest -q tests/python
9 passed in 0.16s

cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 1

./build/flashone_bench
max_abs_diff: 1.86265e-08
standard_attention_ms: 6.69281
flash_attention_tiled_ms: 3.44712
```

### Notes

The benchmark is still reference C++ loops, not oneDNN/AMX. The current value is correctness and harness validation, not final performance.

## 2026-06-25 — Backend seam + score modifiers

### Done

- Checked local oneDNN availability. oneDNN headers/libraries are not installed on this machine.
- Added backend dispatch seam:
  - `AttentionBackendKind::StandardReference`
  - `AttentionBackendKind::FlashTiledReference`
  - `run_attention(...)`
- Added `BlockMask` interface for block-level masking.
- Added `ScoreBiasFn` interface to represent ALiBi/additive score modifiers.
- Updated C++ standard and tiled implementations to share causal/block-mask/bias semantics.
- Updated Python golden reference with block mask and score bias support.
- Added C++ backend/BlockMask/bias tests and Python score modifier tests.
- Added oneDNN BRGEMM integration notes and local environment snapshot.

### Verification

```text
PYTHONPATH=python python3 -m pytest -q tests/python
10 passed in 0.09s

cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 2

./build/flashone_bench
max_abs_diff: 1.86265e-08
standard_attention_ms: 6.64424
flash_attention_tiled_ms: 3.11781
```

## 2026-06-25 — Local oneDNN package extraction + smoke test

### Done

- `sudo` elevation is unavailable from this Feishu runtime, so oneDNN was installed project-locally without using the provided password.
- Downloaded Ubuntu packages with `apt-get download`:
  - `libdnnl-dev=3.1.1-2`
  - `libdnnl3=3.1.1-2`
- Extracted them under `third_party/onednn-local` using `dpkg-deb -x`.
- Added `cmake/FindLocalDnnl.cmake` to find either project-local or system oneDNN.
- Added optional CMake flag `FLASHONE_ENABLE_ONEDNN`.
- Added `flashone_dnnl_smoke`, a oneDNN matmul runtime smoke test.

### Verification

```text
PYTHONPATH=python python3 -m pytest -q tests/python
10 passed in 0.09s

cmake -S . -B build -DFLASHONE_ENABLE_ONEDNN=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 3

./build/flashone_dnnl_smoke
oneDNN smoke matmul max_diff=0
oneDNN runtime version=3.1.1

./build/flashone_bench
max_abs_diff: 1.86265e-08
standard_attention_ms: 5.68878
flash_attention_tiled_ms: 3.01036
```

## 2026-06-26 — TileKernel seam and oneDNN matmul tile

### Done

- Added `include/flashone/tile_kernel.hpp` and `src/flashone/tile_kernel.cpp`.
- Added `TileKernelKind::{Reference, OneDnn}`.
- Added row-major `matmul_tile(...)` seam for `C[M,N] = A[M,K] x B[K,N]`.
- Added oneDNN-backed implementation in `src/flashone/onednn_tile_kernel.cpp`.
- CMake now links oneDNN into `flashone_core` when available and defines `FLASHONE_HAS_ONEDNN`.
- Added `tests/cpp/test_tile_kernel.cpp` to compare oneDNN tile output against reference output.

### Verification

```text
PYTHONPATH=python python3 -m pytest -q tests/python
10 passed in 0.09s

cmake -S . -B build -DFLASHONE_ENABLE_ONEDNN=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 4

./build/flashone_tile_kernel_tests
flashone tile kernel tests passed

./build/flashone_dnnl_smoke
oneDNN smoke matmul max_diff=0
oneDNN runtime version=3.1.1

./build/flashone_bench
max_abs_diff: 1.86265e-08
standard_attention_ms: 5.99141
flash_attention_tiled_ms: 2.99639
```

## 2026-06-26 — QK tile attention uses TileKernel/oneDNN

### Done

- Extended `AttentionOptions` with:
  - `query_block_size`
  - `qk_tile_kernel`
- Added `flash_attention_q_tile(...)`.
- Implemented multi-row Q tile attention:
  - `Q_tile[Qb,H] x K_tile_T[H,Kb] -> score_tile[Qb,Kb]`
  - QK tile multiplication goes through `matmul_tile(...)`
  - online softmax remains in FlashOne code
  - PV accumulation remains scalar/reference for now
- Added `tests/cpp/test_q_tile_attention.cpp`.
- Updated benchmark to report row-tiled, Q-tile reference, and Q-tile oneDNN variants.

### Verification

```text
PYTHONPATH=python python3 -m pytest -q tests/python
10 passed in 0.09s

cmake -S . -B build -DFLASHONE_ENABLE_ONEDNN=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 5

./build/flashone_q_tile_attention_tests
flashone Q-tile attention tests passed

./build/flashone_bench
max_abs_diff_row_tiled: 1.86265e-08
max_abs_diff_q_tile: 1.86265e-08
max_abs_diff_q_tile_onednn: 1.67638e-08
standard_attention_ms: 5.66461
flash_attention_tiled_ms: 3.01477
flash_attention_q_tile_ref_ms: 3.76177
flash_attention_q_tile_onednn_ms: 2.07142
```

## 2026-06-26 — QK+PV tile attention uses oneDNN

### Done

- Extended `AttentionOptions` with `pv_tile_kernel`.
- Added `flash_attention_qk_pv_tile(...)`.
- The new path routes both:
  - QK tile through `options.qk_tile_kernel`
  - PV tile through `options.pv_tile_kernel`
- Online softmax remains explicit in FlashOne code.
- Added `tests/cpp/test_qk_pv_tile_attention.cpp`.
- Updated benchmark to report QK+PV reference and QK+PV oneDNN variants.

### Verification

```text
PYTHONPATH=python python3 -m pytest -q tests/python
10 passed in 0.09s

cmake -S . -B build -DFLASHONE_ENABLE_ONEDNN=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 6

./build/flashone_qk_pv_tile_attention_tests
flashone QK/PV tile attention tests passed

./build/flashone_bench
max_abs_diff_row_tiled: 1.86265e-08
max_abs_diff_q_tile: 1.86265e-08
max_abs_diff_qk_pv_tile: 1.49012e-08
max_abs_diff_q_tile_onednn: 1.67638e-08
max_abs_diff_qk_pv_onednn: 1.86265e-08
standard_attention_ms: 5.65542
flash_attention_tiled_ms: 3.03975
flash_attention_q_tile_ref_ms: 3.73509
flash_attention_qk_pv_tile_ref_ms: 4.37721
flash_attention_q_tile_onednn_ms: 2.08872
flash_attention_qk_pv_onednn_ms: 0.876912
```
## 2026-06-26 TensorFlow Custom Op MVP

- Added a TensorFlow CPU custom op MVP for FlashOne attention: `tensorflow_ops/flashone_attention_op.cc`.
- Added batched/head wrapper around the standalone core: `include/flashone/batched_attention.hpp` and `src/flashone/batched_attention.cpp`. The wrapper accepts flat row-major tensors with shapes `Q=[B,H,M,D]`, `K=[B,H,N,D]`, `V=[B,H,N,Dv]`, `O=[B,H,M,Dv]`.
- Added Python loader package: `python/flashone_tf/`, exposing `flashone_attention(...)` via `tf.load_op_library("build/flashone_tf_attention.so")`.
- CMake now builds `flashone_tf_attention.so` when TensorFlow compile/link flags are available. `flashone_core` is compiled as PIC so it can be linked into the custom op shared object.
- Current op attributes: `causal`, `query_block_size`, `key_block_size`, `use_onednn`. The MVP is CPU-only, inference-only, float32-only, and does not yet register gradients or XLA lowering.
- Verified local TensorFlow environment: TensorFlow `2.21.0`; custom op links to local oneDNN `third_party/onednn-local/usr/lib/x86_64-linux-gnu/libdnnl.so.3` (`3.1.1`).
- Validation passed: `ctest --test-dir build --output-on-failure` -> 6/6 passed; `PYTHONPATH=python python3 -m pytest -q tests/python tests/tensorflow` -> 11 passed.
## 2026-06-26 TensorFlow E2E mini benchmark

- Added TensorFlow E2E benchmark harness: `benchmarks/tensorflow/bench_flashone_tf_e2e.py`. It compares TensorFlow materialized attention, FlashOne custom-op attention, and a minimal decoder block with QKV projection, attention, output projection, FFN, residuals, and layer norms.
- Added E2E correctness test: `tests/tensorflow/test_flashone_tf_e2e.py`, verifying that FlashOne can replace the attention function inside the decoder block.
- Added `tf.load_op_library` caching in `python/flashone_tf/ops.py`; without this, eager-mode benchmark timing included repeated op-library loading.
- Added a simple oneDNN tile primitive cache in `src/flashone/onednn_tile_kernel.cpp`, keyed by `(M,N,K)`, so repeated QK/PV tile shapes reuse engine/stream/matmul primitive descriptors.
- Validation passed: `ctest --test-dir build --output-on-failure` -> 6/6 passed; `PYTHONPATH=python:. python3 -m pytest -q tests/python tests/tensorflow` -> 12 passed.
- C++ microbenchmark after cache: `flash_attention_qk_pv_onednn_ms: 0.717696` for `M=N=128,K=D=64`, vs previous about `0.87ms`; correctness still `max_abs_diff_qk_pv_onednn=1.86265e-08`.
- Eager TensorFlow benchmark, `B=1,H=4,D=32,E=128,q_block=16,k_block=32`: seq64 FlashOne attention `1.565ms` vs TensorFlow attention `2.310ms`, decoder `6.705ms` vs `7.336ms`; seq128 FlashOne attention `2.704ms` vs TensorFlow attention `1.769ms`, decoder `10.805ms` vs `7.860ms`; seq256 FlashOne attention `9.353ms` vs TensorFlow attention `1.122ms`, decoder `15.173ms` vs `10.329ms`.
- Graph-mode sanity (`--graph`) for seq128: TensorFlow attention `0.282851ms`, FlashOne attention `2.752409ms`; decoder `1.409489ms` vs `8.442781ms`. Current Custom Op is usable for E2E replacement but not competitive with TensorFlow graph execution yet.
- Key bottleneck now appears to be Custom Op/kernel overhead plus internal tile copies and per-call allocations; next target should be direct Tensor-backed compute path and buffer reuse, before XLA lowering.

## 2026-06-26 Stage 0 performance baseline + tile sweep harness

- Extended `benchmarks/tensorflow/bench_flashone_tf_e2e.py` with machine-readable outputs:
  - `--output-json`
  - `--output-csv`
- Added tile sweep mode:
  - `--sweep`
  - `--query-blocks`
  - `--key-blocks`
- Generated baseline artifacts under `build/benchmarks/` for eager/graph seq64/128/256 and initial eager tile sweeps.
- Added baseline report: `docs/perf-baseline-2026-06-26.md`.

### Verification

```text
PYTHONPATH=python:. python3 -m pytest -q tests/python tests/tensorflow
12 passed in 2.16s

ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 7

./build/flashone_bench
flash_attention_qk_pv_onednn_ms: 0.729506
max_abs_diff_qk_pv_onednn: 1.86265e-08
```

### Baseline summary

Default TensorFlow benchmark config: `B=1,H=4,D=32,E=128,causal=true,q_block=16,k_block=32`.

- Eager seq64: FlashOne attention `1.103151ms` vs TF attention `2.401837ms`.
- Eager seq128: FlashOne attention `2.601036ms` vs TF attention `1.478697ms`.
- Eager seq256: FlashOne attention `7.303471ms` vs TF attention `0.961287ms`.
- Graph seq64: FlashOne attention `0.692697ms` vs TF attention `0.254572ms`.
- Graph seq128: FlashOne attention `2.132716ms` vs TF attention `0.290322ms`.
- Graph seq256: FlashOne attention `7.646170ms` vs TF attention `0.407576ms`.

Initial tile sweep findings:

- Seq128 eager best: `q_block=32,k_block=32`, FlashOne attention `1.972757ms` (~1.32x faster than default FlashOne seq128).
- Seq256 eager best: `q_block=32,k_block=64`, FlashOne attention `5.276837ms` (~1.38x faster than default FlashOne seq256, still slower than TF).

Next: validate tile heuristic with a wider/repeated sweep, then add default tile selection and continue custom-op overhead audit before XLA Custom Call.

## 2026-06-26 Stage 1A default tile heuristic

- Ran a wider eager tile sweep for seq64/128/256 with `q_block=8,16,32,64` and `k_block=16,32,64,128`.
- Best attention latencies:
  - seq64: `q=64,k=64`, FlashOne `0.631580ms` vs TF `2.503426ms`.
  - seq128: `q=32,k=64`, FlashOne `1.645492ms` vs TF `2.470556ms`.
  - seq256: `q=32,k=64`, FlashOne `5.207241ms` vs TF `2.768849ms`.
- Added `select_tile_sizes(query_tokens, key_tokens)` to `python/flashone_tf/ops.py`.
- Updated `flashone_attention(...)` so omitted block sizes use the heuristic; explicit block sizes still override.
- Updated benchmark CLI defaults so omitted `--query-block/--key-block` use the same heuristic.
- Added TensorFlow tests for heuristic selection and default-wrapper correctness.

### Verification

```text
PYTHONPATH=python:. python3 -m pytest -q tests/python tests/tensorflow
14 passed

ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 7
```

Next: Custom Op data-path audit and graph/XLA minimum custom-call investigation.


## 2026-06-26 Stage 1B custom-op data path audit + strided K view

- Audited TensorFlow Custom Op data path from TF tensors to FlashOne workspace and oneDNN tile kernels.
- Findings:
  - Q/K/V are passed from TensorFlow as raw pointers; no TF-side input copies found.
  - O is allocated once by TensorFlow and written directly by FlashOne.
  - Batched/head wrapper uses `thread_local AttentionWorkspace` and raw pointer offsets.
  - Q and V tile paths are already zero-copy.
  - K tile still had an explicit per-K-block transpose/copy into `ws.k_tile_t`.
  - oneDNN cache lock covered both cache lookup and primitive execution.
- Added `StridedMatmulShape` and `matmul_tile_strided_inplace(...)`.
- QK now uses a strided B view over row-major K (`[D,Kb]` with strides `{1, head_dim}`), removing explicit K transpose.
- Narrowed oneDNN cache mutex scope to lookup/insertion; primitive execution now runs outside the cache mutex.
- Added audit doc: `docs/custom-op-data-path-audit-2026-06-26.md`.

### Verification

```text
PYTHONPATH=python:. python3 -m pytest -q tests/python tests/tensorflow
14 passed

ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 7

./build/flashone_bench
max_abs_diff_qk_pv_onednn=1.86265e-08
flash_attention_qk_pv_onednn_ms=0.772644
```

TensorFlow eager after strided K:

- seq128, tile 32x64: FlashOne `1.798738ms` vs TF `2.422252ms`.
- seq256, tile 32x64: FlashOne `4.807103ms` vs TF `1.490950ms`.

Next: evaluate strided-K vs copied-contiguous-K as a shape-dependent heuristic, and investigate oneDNN memory wrapper/stream wait overhead.

## 2026-06-26 Stage 1B follow-up: QK layout dual path

- Added `QkTileLayout::{CopiedTransposed, StridedK}` to `AttentionOptions`.
- Kept the strided-K path from the data-path audit, and restored the copied/transposed-K path as an explicit alternative.
- Exposed `qk_tile_layout` through the TensorFlow Custom Op attr and Python wrapper:
  - `"strided_k"`
  - `"copied_transposed"`
- Added `select_qk_tile_layout(...)`; current default is always `"strided_k"` for the TensorFlow custom-op path.
- Extended the TensorFlow benchmark with `--qk-tile-layout` / `--qk-layouts` so layout comparisons can be persisted to JSON/CSV.
- Extended workspace tests and TensorFlow tests so both QK layouts are validated against the TensorFlow/reference attention outputs.

### Layout comparison snapshot

C++ microbenchmark, `M=N=128,D=64,V=64,causal=1`:

```text
flash_attention_qk_pv_onednn_ms=0.750128
flash_attention_qk_pv_onednn_copied_k_ms=0.742123
```

TensorFlow eager custom-op benchmark, heuristic tile sizes:

| Shape | Layout | FlashOne attention (ms) | Note |
|---|---|---:|---|
| seq64,D32 | copied_transposed | 2.024155 | slower |
| seq64,D32 | strided_k | 1.512712 | faster |
| seq128,D32 | copied_transposed | 2.441254 | slower |
| seq128,D32 | strided_k | 1.972772 | faster |
| seq256,D32 | copied_transposed | 5.103270 | slower |
| seq256,D32 | strided_k | 4.727749 | faster |
| seq128,D64 | copied_transposed | 3.678657 | slower |
| seq128,D64 | strided_k | 2.201044 | faster |

Interpretation: isolated C++ microbenchmarks can be close and sometimes favor copied K by a tiny margin, but the TensorFlow custom-op path consistently favors strided-K in these samples. Keep copied-K as an experimental/manual path, but default to strided-K.
