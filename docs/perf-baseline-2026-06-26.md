# FlashOne Performance Baseline — 2026-06-26

This baseline freezes the current TensorFlow custom-op path before the next optimization phase.

## Environment

- Project path: `/home/huawei/Desktop/home/xuefenghao/workspace/FlashOne`
- Git commit during run: recorded in JSON outputs as `2fd3b0e`
- TensorFlow: `2.21.0`
- oneDNN: local `third_party/onednn-local`, runtime `3.1.1`
- CPU TensorFlow binary reports AVX2/AVX_VNNI/FMA support available for selected ops; no GPU used.

## Commands

```bash
# Correctness / unit tests
PYTHONPATH=python:. python3 -m pytest -q tests/python tests/tensorflow
ctest --test-dir build --output-on-failure

# C++ microbenchmark
./build/flashone_bench | tee build/benchmarks/cpp_flashone_bench_2026-06-26.txt

# TensorFlow default E2E baseline
for mode in eager graph; do
  graph_flag=""
  if [ "$mode" = graph ]; then graph_flag="--graph"; fi
  for seq in 64 128 256; do
    PYTHONPATH=python:. python3 benchmarks/tensorflow/bench_flashone_tf_e2e.py \
      --seq "$seq" --warmup 3 --repeat 10 $graph_flag \
      --output-json "build/benchmarks/default_${mode}_seq${seq}.json" \
      --output-csv "build/benchmarks/default_${mode}_seq${seq}.csv"
  done
done

# Initial eager tile sweep
PYTHONPATH=python:. python3 benchmarks/tensorflow/bench_flashone_tf_e2e.py \
  --seq 128 --warmup 2 --repeat 5 --sweep \
  --query-blocks 8,16,32 --key-blocks 16,32,64 \
  --output-json build/benchmarks/sweep_eager_seq128.json \
  --output-csv build/benchmarks/sweep_eager_seq128.csv

PYTHONPATH=python:. python3 benchmarks/tensorflow/bench_flashone_tf_e2e.py \
  --seq 256 --warmup 2 --repeat 5 --sweep \
  --query-blocks 8,16,32 --key-blocks 16,32,64,128 \
  --output-json build/benchmarks/sweep_eager_seq256.json \
  --output-csv build/benchmarks/sweep_eager_seq256.csv
```

## Verification

- Python/TensorFlow tests: `12 passed`
- CTest: `7/7 passed`
- C++ benchmark correctness:
  - `max_abs_diff_qk_pv_onednn=1.86265e-08`

## C++ Microbenchmark

Shape: `M=128 N=128 K=64 D=64 causal=1`

| Path | Latency (ms) |
|---|---:|
| standard_attention | 5.64058 |
| flash_attention_tiled | 2.98908 |
| flash_attention_q_tile_ref | 3.21320 |
| flash_attention_qk_pv_tile_ref | 3.14808 |
| flash_attention_q_tile_onednn | 2.00576 |
| flash_attention_qk_pv_onednn | 0.729506 |

## TensorFlow Default Baseline

Shape defaults unless noted: `B=1 H=4 D=32 E=128 causal=true q_block=16 k_block=32`.

### Eager mode

| Seq | TF attention (ms) | FlashOne attention (ms) | FlashOne speedup vs TF | TF decoder (ms) | FlashOne decoder (ms) |
|---:|---:|---:|---:|---:|---:|
| 64 | 2.401837 | 1.103151 | 2.177x | 4.054715 | 6.862019 |
| 128 | 1.478697 | 2.601036 | 0.569x | 8.357491 | 11.177019 |
| 256 | 0.961287 | 7.303471 | 0.132x | 4.576654 | 10.928397 |

### Graph mode (`tf.function(jit_compile=False)`)

| Seq | TF attention (ms) | FlashOne attention (ms) | FlashOne speedup vs TF | TF decoder (ms) | FlashOne decoder (ms) |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.254572 | 0.692697 | 0.368x | 0.907308 | 1.370904 |
| 128 | 0.290322 | 2.132716 | 0.136x | 1.151255 | 5.968866 |
| 256 | 0.407576 | 7.646170 | 0.053x | 2.422011 | 9.801089 |

## Initial Tile Sweep Findings

### Seq128 eager

Best FlashOne attention latency in this sweep:

- `q_block=32, k_block=32`
- FlashOne attention: `1.972757ms`
- TF attention in same run: `2.510026ms`
- Speedup vs TF in same run: `1.272x`

Compared with default seq128 eager FlashOne latency `2.601036ms`, this is about `1.32x` faster.

### Seq256 eager

Best FlashOne attention latency in this sweep:

- `q_block=32, k_block=64`
- FlashOne attention: `5.276837ms`
- TF attention in same run: `2.541132ms`
- Speedup vs TF in same run: `0.482x`

Compared with default seq256 eager FlashOne latency `7.303471ms`, this is about `1.38x` faster, but still slower than TensorFlow.

## Interpretation

1. The current C++ oneDNN QK+PV tile path remains strong: `~0.73ms` for the reference `128x128x64` shape.
2. The TensorFlow eager custom-op path can beat TensorFlow attention at small shapes, but default tile sizes are not optimal for seq128/seq256.
3. Tile tuning is immediately useful: seq128 improves from `2.60ms` to `1.97ms`; seq256 improves from `7.30ms` to `5.28ms`.
4. Graph mode remains the strategic gap. TensorFlow graph attention is much faster than the current custom-op graph wrapper, so XLA lowering / custom call is still required.

## Next Actions

1. Add a small tile heuristic in the Python/TF wrapper or C++ op defaults:
   - seq64: keep current `q=16,k=32` until a seq64 sweep says otherwise.
   - seq128: prefer `q=32,k=32`.
   - seq256: prefer `q=32,k=64`.
2. Run a wider seq64/128/256 sweep with more repeats to reduce timing noise.
3. Audit custom-op data path for remaining wrapper/scratchpad/memory descriptor overhead.
4. Start the XLA Custom Call minimum viable path after the above default heuristic is validated.

## Follow-up Wider Eager Sweep and Default Tile Heuristic

A wider eager sweep was run with `q_block=8,16,32,64`, `k_block=16,32,64,128`, `warmup=3`, `repeat=8`.

Artifacts:

- `build/benchmarks/sweep2_eager_seq64.json/csv`
- `build/benchmarks/sweep2_eager_seq128.json/csv`
- `build/benchmarks/sweep2_eager_seq256.json/csv`

Best FlashOne attention latencies:

| Seq | Best q_block | Best k_block | FlashOne attention (ms) | TF attention in same run (ms) | Speedup vs TF |
|---:|---:|---:|---:|---:|---:|
| 64 | 64 | 64 | 0.631580 | 2.503426 | 3.964x |
| 128 | 32 | 64 | 1.645492 | 2.470556 | 1.501x |
| 256 | 32 | 64 | 5.207241 | 2.768849 | 0.532x |

Implemented Python wrapper heuristic in `python/flashone_tf/ops.py`:

- `seq <= 64`: `q_block=64,k_block=64` capped by actual query/key length.
- otherwise: `q_block=32,k_block=64` capped by actual query/key length.
- Explicit `query_block_size` / `key_block_size` still override the heuristic.
- Dynamic token dimensions require explicit block sizes.

This heuristic applies to Python wrapper calls. The low-level TensorFlow op attributes still keep their explicit registered defaults (`16/32`) for ABI stability; the benchmark harness now uses the wrapper heuristic when `--query-block/--key-block` are omitted.

