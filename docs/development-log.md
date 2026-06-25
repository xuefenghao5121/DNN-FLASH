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
