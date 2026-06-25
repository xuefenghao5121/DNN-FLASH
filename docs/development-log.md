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
