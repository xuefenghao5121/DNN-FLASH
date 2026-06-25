# FlashOne

**FlashOne = Flash Attention on oneDNN**

FlashOne is a CPU-side FlexAttention prototype based on TensorFlow/XLA Custom Call and oneDNN-style BRGEMM tiling. The project starts with a standalone C++/Python reference implementation so the math, tiling policy, correctness tests, and benchmark harness can be validated before integrating TensorFlow, XLA, and oneDNN internals.

## Goals

- Build Flash-style tiled attention for CPU without materializing the full `N x N` score/probability matrix.
- Validate online softmax recurrence and block-level mask skipping.
- Prepare a bridge from XLA epilogues (`score_mod`) to oneDNN post-ops.
- Target x64 AMX/AVX-512 first, then ARM SVE/Kunpeng 930.

## Current milestone

Month 1 / Week 1 scaffold:

- C++ reference standard attention
- C++ Flash-style tiled attention
- Python/Numpy golden implementation
- Correctness tests
- Minimal benchmark harness
- Architecture and requirements docs

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/flashone_bench
```

## Layout

```text
include/flashone/     Public C++ headers
src/flashone/         C++ implementation
python/flashone/      Python reference/golden code
tests/cpp/            C++ tests
tests/python/         Python tests
benchmarks/           Benchmark entry points
docs/                 Design, requirements, project plan
```

## Roadmap

See `docs/project-plan.md` for the full two-phase plan:

1. Phase 1: FlexAttention on CPU
2. Phase 2: FlexInteraction for recommendation feature interaction
