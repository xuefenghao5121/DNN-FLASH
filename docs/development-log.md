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
