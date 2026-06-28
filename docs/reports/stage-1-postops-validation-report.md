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

The following numbers are from `benchmarks/results/stage-1-postops/cpp-qk-postops.json`, warmup=3, repeat=15. `time_ms` is kept as mean for backward compatibility; the table uses median as the primary comparison metric and includes p90/stddev to expose benchmark noise.

| Shape | Score mod | Ref median ms | oneDNN median ms | Ratio ref/oneDNN | oneDNN p90 ms | oneDNN stddev ms | Max diff |
|---|---|---:|---:|---:|---:|---:|---:|
| B=1,H=1,M=N=64,D=32 | none | 0.049367 | 0.028469 | 1.73x | 0.175147 | 0.067248 | 0 |
| B=1,H=1,M=N=64,D=32 | scale | 0.030861 | 0.020976 | 1.47x | 0.026161 | 0.004426 | 0 |
| B=1,H=1,M=N=64,D=32 | scale + same-shape bias | 0.034133 | 0.020948 | 1.63x | 0.021634 | 0.000567 | 0 |
| B=1,H=1,M=N=128,D=64 | none | 0.264887 | 0.036660 | 7.23x | 0.039364 | 0.001378 | 1e-6 |
| B=1,H=1,M=N=128,D=64 | scale | 0.311027 | 0.025246 | 12.32x | 0.027642 | 0.001308 | 0 |
| B=1,H=1,M=N=128,D=64 | scale + same-shape bias | 0.266708 | 0.026077 | 10.23x | 0.029837 | 0.001493 | 0 |
| B=1,H=4,M=N=128,D=64 | none | 1.030267 | 0.096562 | 10.67x | 0.099559 | 0.003383 | 1e-6 |
| B=1,H=4,M=N=128,D=64 | scale | 1.037449 | 0.123863 | 8.38x | 0.138974 | 0.007708 | 0 |
| B=1,H=4,M=N=128,D=64 | scale + same-shape bias | 1.080860 | 0.121611 | 8.89x | 0.132892 | 0.005760 | 0 |

Interpretation:

- Reusing oneDNN memory wrappers and rebinding their data handles reduces wrapper-construction overhead for several score-mod cases, especially 64x64 scale/scale+bias and 128x128 H1 scale.
- For small 64x64 tiles, oneDNN is now competitive after primitive-cache and memory-wrapper hardening, but the `none` case still shows high tail noise (`p90`/`stddev`), so median should be preferred over a single mean when judging micro-optimizations.
- For 128x128x64 tiles, oneDNN matmul is clearly faster at this isolated QK score-tile level across none/scale/scale+bias.
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

Proceed to the next short RuntimePlan/oneDNN integration hardening step before XLA ABI work. Two bounded options remain:

1. **Stream wait granularity evaluation**: measure whether moving wait out of per-tile QK calls is safe/valuable in the current synchronous API contract.
2. **Cache observability**: add debug counters or benchmark-only instrumentation for primitive-cache hits/misses and memory-handle rebinding cost.

My recommendation: evaluate stream wait granularity next, but keep it as a measurement/design step first. The current public QK tile API writes caller-owned `score` and returns synchronously, so any deferred wait must either stay internal to batched/full attention call boundaries or preserve the existing synchronous tile contract.
