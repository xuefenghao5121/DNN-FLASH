# Document Status

> Historical/MVP design note.
> Current system design baseline is `docs/system-design.md`.
> This document records the early standalone-core bootstrap rationale and must not be used as the main roadmap for new implementation work.
> When this document conflicts with `docs/system-design.md`, follow `docs/system-design.md`.

# FlashOne Design

## Design principle

Start with a small, verifiable, standalone attention core. Do not hide algorithmic uncertainty behind TensorFlow/XLA/oneDNN integration. Once the recurrence, masks, tests, and benchmark harness are stable, replace reference inner loops with oneDNN BRGEMM/JIT kernels and connect the runtime to XLA Custom Call.

## Layers

```text
Python API / tests
  └─ Numpy golden attention
C++ standalone core
  ├─ Standard attention reference
  ├─ Flash-style tiled online softmax attention
  └─ Benchmark/test harness
Future integration
  ├─ TensorFlow Custom Op
  ├─ XLA Custom Call bridge
  └─ oneDNN BRGEMM primitive/JIT backend
```

## Tensor convention

Initial C++ core uses flattened row-major tensors:

- `Q`: `[M, K]`
- `K`: `[N, K]`
- `V`: `[N, D]`
- `O`: `[M, D]`

The first version is single-head. Batch/head dimensions will be added as outer loops after the core recurrence is stable.

## Online softmax recurrence

For every query row `i`, process key/value blocks. For each block compute local scores:

```text
s_j = dot(Q_i, K_j) * scale + bias(i, j)
```

Maintain:

- `m`: running row maximum
- `l`: running denominator sum
- `acc[d]`: running numerator sum for output dimension `d`

When a block has local max `m_new = max(m, block_max)`, rescale previous state:

```text
alpha = exp(m - m_new)
l = l * alpha + sum_j exp(s_j - m_new)
acc[d] = acc[d] * alpha + sum_j exp(s_j - m_new) * V[j, d]
m = m_new
```

Final output:

```text
O[i, d] = acc[d] / l
```

This is the CPU-side FlashAttention math foundation.

## Mask strategy

Current MVP:

- no mask
- causal mask: skip `j > i`

Future BlockMask:

- block-level skip before QK computation
- dense in-block predicate for boundary tiles
- sliding window, prefix, custom block sparse masks

## Future oneDNN mapping

- QK tile = BRGEMM over K dimension
- score_mod = post-op chain after QK accumulation
- online softmax = row-wise reduction + register/local buffer update
- PV tile accumulation = BRGEMM batch-reduce style accumulation
- primitive cache keyed by shape, dtype, ISA, mask type, score_mod signature

## Development gates

1. G0: standalone reference + tiled implementation pass correctness.
2. G1: replace selected inner loops with oneDNN BRGEMM or microkernel wrappers.
3. G2: integrate TensorFlow Custom Op.
4. G3: integrate XLA Custom Call and verify HLO is not split into standard attention.
5. G4: benchmark AMX/AVX-512/ARM SVE and decide Phase 2 readiness.
