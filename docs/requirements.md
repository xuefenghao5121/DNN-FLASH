# FlashOne Requirements

## Mission

FlashOne provides a CPU-side FlexAttention implementation with FlashAttention-style memory behavior and oneDNN-oriented kernel design.

## MVP Scope — Month 1 Week 1

The first development slice intentionally avoids TensorFlow/XLA/oneDNN integration. It establishes the executable math and test harness.

### Functional requirements

- FR-001: Provide standard scaled dot-product attention reference for `[M, K] x [N, K] x [N, D]` tensors.
- FR-002: Provide Flash-style tiled attention using online softmax recurrence.
- FR-003: Support causal mask.
- FR-004: Support additive score bias hook in the internal API design, even if the first C++ implementation only uses causal/none.
- FR-005: Provide Python/Numpy golden implementation for correctness comparison.
- FR-006: Provide C++ and Python smoke/correctness tests.
- FR-007: Provide a minimal benchmark executable.

### Non-functional requirements

- NFR-001: No full probability matrix materialization in the Flash-style implementation.
- NFR-002: Correctness tolerance for float32 reference path: max absolute error <= `1e-4` for small test shapes.
- NFR-003: Project must build with system CMake and GCC/Clang using C++17.
- NFR-004: Python tests must run with `pytest` and `numpy` only.

## Phase 1 target requirements

- P1-001: TensorFlow Custom Op path.
- P1-002: XLA Custom Call path.
- P1-003: oneDNN BRGEMM-backed QK/PV kernels.
- P1-004: score_mod lowering to post-op chain: scale, bias, ALiBi, tanh gate.
- P1-005: BlockMask tile-level skipping.
- P1-006: bf16 primary path, f32 validation path.
- P1-007: x64 AMX/AVX-512 target and ARM SVE validation.

## Acceptance for first commit

- `PYTHONPATH=python python3 -m pytest -q tests/python` passes.
- `cmake -S . -B build && cmake --build build -j` passes.
- `ctest --test-dir build --output-on-failure` passes.
- `./build/flashone_bench` runs and prints timing for standard and tiled implementations.
