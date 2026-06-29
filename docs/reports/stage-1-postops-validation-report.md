# FlashOne Stage 1 QK Post-ops Validation Report

> Date: 2026-06-29 (updated)
> Scope: Stage 1.2 / Stage 1.3 minimal QK score tile validation + Stage 1.10 cache observability.
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

The following numbers are from `benchmarks/results/stage-1-postops/cpp-qk-postops.json`, warmup=3, repeat=15, generated with `--include-deferred-wait`. `time_ms` is kept as mean for backward compatibility; the table uses median as the primary comparison metric and includes p90/stddev to expose benchmark noise. The default public tile path remains `wait per tile`; `defer per record` is an internal benchmark mode that batches oneDNN stream waits until all B/H tiles in one record have been submitted.

| Shape | Score mod | Sync mode | Ref median ms | oneDNN median ms | Ratio ref/oneDNN | oneDNN p90 ms | oneDNN stddev ms | Max diff |
|---|---|---|---:|---:|---:|---:|---:|---:|
| B=1,H=1,M=N=64,D=32 | none | wait per tile | 0.084556 | 0.033219 | 2.55x | 0.166788 | 0.052984 | 0 |
| B=1,H=1,M=N=64,D=32 | none | defer per record | 0.084556 | 0.028922 | 2.92x | 0.036331 | 0.005555 | 0 |
| B=1,H=1,M=N=64,D=32 | scale | wait per tile | 0.034507 | 0.032366 | 1.07x | 0.043353 | 0.005674 | 0 |
| B=1,H=1,M=N=64,D=32 | scale | defer per record | 0.034507 | 0.030903 | 1.12x | 0.037659 | 0.004365 | 0 |
| B=1,H=1,M=N=64,D=32 | scale + same-shape bias | wait per tile | 0.049052 | 0.032101 | 1.53x | 0.038834 | 0.003478 | 0 |
| B=1,H=1,M=N=64,D=32 | scale + same-shape bias | defer per record | 0.049052 | 0.031073 | 1.58x | 0.033925 | 0.002001 | 0 |
| B=1,H=1,M=N=128,D=64 | none | wait per tile | 0.261333 | 0.024851 | 10.52x | 0.029656 | 0.002026 | 1e-6 |
| B=1,H=1,M=N=128,D=64 | none | defer per record | 0.261333 | 0.024366 | 10.73x | 0.025012 | 0.000658 | 1e-6 |
| B=1,H=1,M=N=128,D=64 | scale | wait per tile | 0.259253 | 0.024669 | 10.51x | 0.029237 | 0.001871 | 0 |
| B=1,H=1,M=N=128,D=64 | scale | defer per record | 0.259253 | 0.024257 | 10.69x | 0.025980 | 0.002521 | 0 |
| B=1,H=1,M=N=128,D=64 | scale + same-shape bias | wait per tile | 0.269568 | 0.024632 | 10.94x | 0.027723 | 0.001827 | 0 |
| B=1,H=1,M=N=128,D=64 | scale + same-shape bias | defer per record | 0.269568 | 0.023556 | 11.44x | 0.024384 | 0.000664 | 0 |
| B=1,H=4,M=N=128,D=64 | none | wait per tile | 1.036479 | 0.098304 | 10.54x | 0.102125 | 0.003814 | 1e-6 |
| B=1,H=4,M=N=128,D=64 | none | defer per record | 1.036479 | 0.095963 | 10.80x | 0.100216 | 0.003871 | 1e-6 |
| B=1,H=4,M=N=128,D=64 | scale | wait per tile | 1.039304 | 0.097234 | 10.69x | 0.103486 | 0.003261 | 0 |
| B=1,H=4,M=N=128,D=64 | scale | defer per record | 1.039304 | 0.095443 | 10.89x | 0.097388 | 0.001523 | 0 |
| B=1,H=4,M=N=128,D=64 | scale + same-shape bias | wait per tile | 1.070353 | 0.097462 | 10.98x | 0.104051 | 0.004722 | 0 |
| B=1,H=4,M=N=128,D=64 | scale + same-shape bias | defer per record | 1.070353 | 0.098922 | 10.82x | 0.101641 | 0.004377 | 0 |

Interpretation:

- Deferred stream wait is correctness-stable in this isolated benchmark (`max_abs_diff <= 1e-6`) and usually reduces p90/stddev, especially for small 64x64 tiles and H=4 scale cases.
- Median gains are modest because this benchmark submits only one to four QK tiles per record; larger batched/head call boundaries may expose more wait-amortization opportunity.
- The public `qk_score_tile_inplace(...)` contract remains synchronous. Deferred wait is only exposed through an internal options API and must be explicitly followed by `qk_score_tile_wait_for_onednn()` before reading caller-owned score buffers.
- Same-shape additive bias post-op remains viable; no correctness instability was observed.
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

## Stage 1.10 Cache Observability

Stage 1.10 adds cache observability counters to the oneDNN QK post-ops path:

- `QkScoreTileCacheStats` struct exposed via `qk_score_tile_internal.hpp`.
- `qk_score_tile_get_cache_stats()` / `qk_score_tile_reset_cache_stats()` internal API.
- Counters: `primitive_cache_hits`, `primitive_cache_misses`, `memory_handle_rebinds`, `immediate_waits`, `deferred_waits`, `cache_size`.
- Benchmark schema upgraded to `v4`; JSON records now include a `cache_stats` object; CSV includes the six new columns.
- Test `test_cache_observability_counters` verifies hit/miss/rebind/wait counters across first call, second call, and reset.

Benchmark cache stats observations (warmup=3, repeat=15, `--include-deferred-wait`):

- oneDNN rows show 0 misses during timed repeats (cache populated during warmup).
- 15 hits per oneDNN record (one tile per repeat, 15 repeats).
- 15 handle rebinds per oneDNN record (one rebind per tile execute).
- `wait_after_execute` rows: 15 immediate_waits, 0 deferred_waits.
- `defer_until_record_end` rows: 0 immediate_waits, 15 deferred_waits (wait happens once per record via `qk_score_tile_wait_for_onednn()`).
- `cache_size` grows as new shape/score_mod combinations are encountered.

These counters confirm that the primitive cache and memory wrapper reuse from Stage 1.5–1.8 are functioning correctly during benchmark runs, and they provide a foundation for diagnosing whether future performance changes come from cache behavior, handle rebinding, or stream synchronization.

## Next-stage Recommendation

Stage 1.10 adds cache observability instrumentation:

- `QkScoreTileCacheStats` struct exposed via internal header, reporting `primitive_cache_hits`, `primitive_cache_misses`, `memory_handle_rebinds`, `immediate_waits`, `deferred_waits`, and `cache_size`.
- `qk_score_tile_get_cache_stats()` / `qk_score_tile_reset_cache_stats()` provide query and reset without affecting the primitive cache itself.
- Benchmark schema upgraded to `v4` with per-record `cache_stats` block.
- Test `test_cache_observability_counters` verifies hit/miss/rebind/wait counter accuracy across first-call, second-call, and reset.

Cache stats observations from v4 benchmark (oneDNN rows, wait-per-tile, warmup=3, repeat=15):

| Shape | Score mod | Cache hits | Cache misses | Rebinds | Immediate waits | Cache size |
|---|---|---:|---:|---:|---:|---:|
| 64x64 D32 | none | 15 | 0 | 15 | 15 | 1 |
| 64x64 D32 | scale | 15 | 0 | 15 | 15 | 2 |
| 64x64 D32 | scale+bias | 15 | 0 | 15 | 15 | 3 |
| 128x128 D64 H1 | none | 15 | 0 | 15 | 15 | 4 |
| 128x128 D64 H4 | none | 60 | 0 | 60 | 60 | 5 |

Key findings:
- After warmup, all timed repeats hit the primitive cache (0 misses), confirming the cache key is stable across tiles of the same shape/score_mod.
- `handle_rebinds` equals the number of oneDNN execute calls, confirming `set_data_handle` is called once per tile.
- `immediate_waits` equals execute calls for wait-per-tile mode; `deferred_waits` is 0 in that mode.
- `cache_size` grows incrementally as new shape/score_mod combinations are encountered.

Do not make deferred wait the default public QK tile behavior. If deferred wait is promoted later, it should be used only inside a higher-level internal call boundary that owns all submitted score buffers and can guarantee a final wait before softmax/PV reads them.

Next steps: 1) Evaluate whether per-tile `set_data_handle` overhead is measurable now that hit/miss is observable; 2) Consider batched handle-rebind+execute patterns if rebind cost is significant; 3) Prepare for XLA CustomCall ABI exploration.
