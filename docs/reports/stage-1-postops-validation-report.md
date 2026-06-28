# FlashOne Stage 1 QK Post-ops Validation Report

> Date: 2026-06-28
> Scope: Stage 1.2 / Stage 1.3 minimal QK score tile validation.
> Boundary: C++ QK score-tile level only; this does **not** claim TensorFlow graph/XLA speedup.

## Environment

- Git commit: `5c6e98c` at benchmark generation time.
- oneDNN version: `3.12.1`.
- Benchmark level: `cpp_qk_score_tile`.
- Machine-readable outputs:
  - `benchmarks/results/stage-1-postops/onednn-postops-capability.json`
  - `benchmarks/results/stage-1-postops/cpp-qk-postops.json`
  - `benchmarks/results/stage-1-postops/cpp-qk-postops.csv`

## Capability Probe Results

Stage 1.1 confirmed the following oneDNN `dnnl::matmul + post_ops` capability boundary:

| Capability | Result | Stage 1 Interpretation |
|---|---:|---|
| `matmul.scale` | supported | use eltwise linear post-op for scale |
| `matmul.binary_add_same_shape` | supported | use binary add post-op with same-shape bias tile |
| `matmul.binary_add_broadcast_row` | supported by probe | still fallback in RuntimePlan until design promotes it |
| `matmul.binary_add_broadcast_col` | supported by probe | still fallback in RuntimePlan until design promotes it |
| `matmul.eltwise_linear` | supported | scale path evidence |
| `ukernel_brgemm.post_ops_boundary` | intentionally unsupported | BRGEMM remains baseline/candidate context, not Stage 1 post-op path |
| `ukernel_transform.supported_out_ld` | supported for 16/32/48/64 | layout evidence only |

## Implementation Summary

Stage 1.2/1.3 adds a minimal internal QK score tile path:

- `src/flashone/qk_score_tile.cpp`
- `src/flashone/qk_score_tile_internal.hpp`
- `tests/cpp/test_qk_score_tile.cpp`
- `benchmarks/bench_qk_postops.cpp`

The path consumes `RuntimePlan` / `ScoreModPlan` instead of making backend decisions locally:

```text
RuntimePlan -> qk_score_tile_inplace -> dnnl::matmul + post_ops
```

Supported fast-path score_mod semantics:

- `none`: oneDNN matmul without post-ops.
- `scale:f32`: `dnnl::post_ops.append_eltwise(eltwise_linear, scale, 0)`.
- `scale_additive_bias:same_shape_tile:f32`: scale post-op followed by binary add post-op.

Fallback behavior:

- Unsupported/broadcast bias remains reference fallback with observable `FallbackReason`.
- oneDNN exceptions fall back to reference QK plus FlashOne post-op application.
- Causal/boundary mask remains outside this path and is still owned by FlashOne epilogue / mask tile generator design.

## Correctness Results

C++ tests cover:

- scale post-op correctness vs reference.
- scale + same-shape additive bias correctness vs reference.
- broadcast bias fallback correctness and debug reason visibility.

Observed benchmark correctness:

- All `cpp-qk-postops` benchmark records have `max_abs_diff <= 1e-6`.
- oneDNN `scale` and `scale_additive_bias` records used post-ops with `fallback_reason=none`.

## Performance Results

The following numbers are from `benchmarks/results/stage-1-postops/cpp-qk-postops.json`, warmup=2, repeat=5.

| Shape | Score mod | Reference ms | oneDNN matmul ms | Ratio ref/oneDNN | Max diff |
|---|---|---:|---:|---:|---:|
| B=1,H=1,M=N=64,D=32 | none | 0.053614 | 0.161356 | 0.33x | 0 |
| B=1,H=1,M=N=64,D=32 | scale | 0.034252 | 0.077566 | 0.44x | 0 |
| B=1,H=1,M=N=64,D=32 | scale + same-shape bias | 0.045339 | 0.046386 | 0.98x | 0 |
| B=1,H=1,M=N=128,D=64 | none | 0.450281 | 0.072427 | 6.22x | 1e-6 |
| B=1,H=1,M=N=128,D=64 | scale | 0.263794 | 0.037582 | 7.02x | 0 |
| B=1,H=1,M=N=128,D=64 | scale + same-shape bias | 0.270311 | 0.041573 | 6.50x | 0 |
| B=1,H=4,M=N=128,D=64 | none | 1.034173 | 0.129057 | 8.01x | 1e-6 |
| B=1,H=4,M=N=128,D=64 | scale | 1.040615 | 0.126629 | 8.22x | 0 |
| B=1,H=4,M=N=128,D=64 | scale + same-shape bias | 1.077850 | 0.134453 | 8.02x | 0 |

Interpretation:

- For small 64x64 tiles, primitive construction / memory wrapper overhead dominates and oneDNN is not consistently faster.
- For 128x128x64 tiles, oneDNN matmul is clearly faster at this isolated QK score-tile level.
- Same-shape additive bias post-op appears viable; no correctness instability was observed.
- These numbers should not be projected to full attention, TensorFlow eager, or XLA graph paths because online softmax, PV accumulation, TensorFlow op overhead, primitive caching, and stream synchronization are not represented fully here.

## Fallback Summary

- `reference` benchmark rows intentionally use `DebugForcedReference`.
- `onednn_matmul` rows for supported Stage 1 score_mods have `fallback_reason=none`.
- Broadcast-row bias is still fallback in `RuntimePlan` even though capability probe says oneDNN can compile/run it; this preserves the frozen Stage 1 boundary.
- BRGEMM ukernel post-ops remain unsupported for Stage 1 and are not used by the QK post-op path.

## Interpretation Boundary

This report validates that oneDNN JIT/post-ops can carry Stage 1 QK score_mod semantics for f32 QK score tiles:

```text
QK
QK * scale
QK * scale + same_shape_bias_tile
```

It does not validate:

- full online softmax recurrence changes;
- PV accumulation changes;
- TensorFlow graph/XLA speedup;
- bf16/AMX production behavior;
- broadcast bias promotion;
- BRGEMM ukernel post-op main path.

## Next-stage Recommendation

Proceed to Stage 1.4 report/gate closure and then decide between two next steps:

1. **RuntimePlan integration hardening**: add primitive/cache-key layering for this QK post-op path and reduce per-call primitive/memory wrapper overhead.
2. **XLA CustomCall minimum ABI investigation**: only after RuntimePlan semantics and benchmark/report schema remain stable.

My recommendation: do one short RuntimePlan/oneDNN integration hardening pass first, because current C++ numbers show overhead sensitivity at small tiles and the implementation still constructs oneDNN engine/stream/primitive wrappers per call.
