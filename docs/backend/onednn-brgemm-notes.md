# oneDNN / BRGEMM Integration Notes

## Current local environment check

Checked on 2026-06-25:

- C++ compiler: GCC 13.3.0
- CMake: 4.3.1
- TBB headers are present under `/usr/include/oneapi/tbb`
- System oneDNN development headers/libraries were not installed, but Ubuntu packages were downloaded and extracted locally under `third_party/onednn-local`. The optional CMake path now builds `flashone_dnnl_smoke` and verifies oneDNN matmul at runtime.

The repository still keeps oneDNN integration behind a backend seam so the reference path continues to build when oneDNN is unavailable.

## Why a backend seam now

The standalone tiled attention core already exposes the algorithmic replacement point:

```text
run_attention(kind, q, k, v, shape, options)
```

Current backend kinds:

- `StandardReference`
- `FlashTiledReference`

Future backend kinds:

- `OneDnnBrgemmReferenceLayout`
- `OneDnnAmxBf16`
- `OneDnnAvx512F32`
- `SveReferenceOrACLE`

The intent is to preserve tests and API while replacing only the inner QK/PV tile kernels.

## Planned oneDNN build

Recommended source build shape:

```bash
git clone https://github.com/oneapi-src/oneDNN.git third_party/oneDNN
cmake -S third_party/oneDNN -B build-onednn \
  -DCMAKE_BUILD_TYPE=Release \
  -DDNNL_CPU_RUNTIME=THREADPOOL \
  -DDNNL_BUILD_TESTS=OFF \
  -DDNNL_BUILD_EXAMPLES=OFF \
  -DDNNL_EXPERIMENTAL_UKERNEL=ON
cmake --build build-onednn -j
```

Exact flags may need adjustment for the oneDNN version available on the target machine.

## FlashOne mapping

For each query tile and key tile:

1. QK tile:
   - Reference today: nested loop dot product
   - oneDNN target: BRGEMM matmul/microkernel
2. score_mod:
   - Reference today: `ScoreBiasFn`, causal predicate, `BlockMask`
   - oneDNN target: post-op chain + in-block predicate
3. online softmax:
   - Reference today: row-wise max/sum recurrence in C++
   - oneDNN target: local row reduction + register/L1 resident state
4. PV accumulation:
   - Reference today: loop over values
   - oneDNN target: BRGEMM batch-reduce style accumulation

## Current CMake integration

`FLASHONE_ENABLE_ONEDNN` is available and defaults to `ON`. If `dnnl.hpp`/`dnnl.h` plus `libdnnl` are found, FlashOne builds `flashone_dnnl_smoke`; otherwise it prints a status message and skips that target.

## Immediate next coding step

Introduce a `TileKernel` abstraction for QK and PV tiles, then implement a first oneDNN-backed dense matmul tile smoke before attempting full online-softmax integration.

## TileKernel seam — 2026-06-26

A first `TileKernel` seam now exists:

```cpp
enum class TileKernelKind {
    Reference,
    OneDnn,
};

std::vector<float> matmul_tile(TileKernelKind kind,
                               const std::vector<float>& a,
                               const std::vector<float>& b,
                               const MatmulShape& shape);
```

Semantics:

- Row-major `C[M,N] = A[M,K] x B[K,N]`.
- Reference implementation is always available.
- oneDNN implementation is compiled when `FLASHONE_HAS_ONEDNN` is defined by CMake.

Current oneDNN implementation uses `dnnl::matmul`. This is not the final BRGEMM ukernel path, but it proves that QK/PV tile multiplication can be routed through oneDNN behind a stable interface.

Next replacement step:

1. Introduce a tile policy that extracts Q/K/V tile views without copying where possible.
2. Replace the QK dot loop in `flash_attention_tiled` with `matmul_tile` for multi-row query tiles.
3. Keep online softmax in FlashOne code between QK and PV.
4. Replace PV accumulation with `matmul_tile` once the weighted-probability tile is materialized in a small local buffer.

## QK Tile Attention Pipeline — 2026-06-26

`flash_attention_q_tile(...)` now computes QK in multi-row query tiles:

```text
Q_tile[Qb, H] x K_tile_T[H, Kb] -> score_tile[Qb, Kb]
```

The QK tile multiplication is routed through `matmul_tile(options.qk_tile_kernel, ...)`, so it can use either:

- `TileKernelKind::Reference`
- `TileKernelKind::OneDnn`

Current pipeline:

1. Copy a small Q tile into row-major local buffer.
2. Transpose K block into `[H, Kb]` local buffer.
3. Call `matmul_tile` to compute raw QK score tile.
4. Apply scale, causal mask, `BlockMask`, and `ScoreBiasFn` in FlashOne code.
5. Run per-query online softmax recurrence.
6. Accumulate `weight * V` using scalar/reference loops.

This proves the first real backend substitution point: **QK tile computation can now use oneDNN inside attention** while preserving correctness with causal mask, block mask, and score bias.

Current benchmark on local machine (`M=N=128,H=D=64,causal,key_block=32,query_block=16`):

```text
max_abs_diff_q_tile_onednn: 1.67638e-08
flash_attention_q_tile_ref_ms: 3.76177
flash_attention_q_tile_onednn_ms: 2.07142
```

Next step:

- Add a PV tile path: materialize a small `P_tile[Qb,Kb]`, then use `matmul_tile(P_tile, V_tile)` for output accumulation.
- After both QK and PV are routed through `TileKernel`, replace `dnnl::matmul` with lower-level BRGEMM/ukernel if the installed oneDNN version exposes the needed APIs.

## QK + PV Tile Attention Pipeline — 2026-06-26

`flash_attention_qk_pv_tile(...)` now routes both main matrix multiplications through `TileKernel`:

```text
Q_tile[Qb,H] x K_tile_T[H,Kb] -> score_tile[Qb,Kb]
P_tile[Qb,Kb] x V_tile[Kb,D] -> pv_tile[Qb,D]
```

Pipeline:

1. Compute raw QK score tile through `options.qk_tile_kernel`.
2. Apply scale, causal mask, `BlockMask`, and `ScoreBiasFn`.
3. Compute online-softmax local weights in `P_tile`, using the updated running max.
4. Compute `P_tile x V_tile` through `options.pv_tile_kernel`.
5. Rescale historical accumulator by `exp(old_max - new_max)` and add the PV tile.
6. Update running denominator and final normalize at the end of all K blocks.

This gives the first end-to-end FlashOne CPU pipeline where both QK and PV can use oneDNN while online softmax remains explicit in FlashOne code.

Current local benchmark (`M=N=128,H=D=64,causal,key_block=32,query_block=16`):

```text
max_abs_diff_qk_pv_onednn: 1.86265e-08
flash_attention_qk_pv_onednn_ms: 0.876912
standard_attention_ms: 5.65542
flash_attention_tiled_ms: 3.03975
flash_attention_q_tile_onednn_ms: 2.08872
```

Interpretation:

- QK-only oneDNN proves the first backend substitution point.
- QK+PV oneDNN is much faster in this small benchmark because the expensive `weight * V` accumulation also leaves scalar loops.
- The current implementation still copies/transposes tiles and creates oneDNN primitives per call; primitive caching and buffer reuse remain major next steps.

Next steps:

1. Add primitive/cache reuse inside oneDNN tile kernel keyed by `(M,N,K)`.
2. Add reusable scratch buffers for Q/K/V/P tiles to reduce per-block allocation.
3. Benchmark shape sweep: sequence length, head dim, value dim, block sizes.
## TensorFlow Custom Op linkage note

The first TensorFlow integration is a CPU custom op, not XLA lowering yet. The shared library target is `flashone_tf_attention.so`; it reuses `flashone_core` and links against the local oneDNN package extracted under `third_party/onednn-local`. At runtime, `ldd build/flashone_tf_attention.so` resolves `libdnnl.so.3` to `third_party/onednn-local/usr/lib/x86_64-linux-gnu/libdnnl.so.3`.

This keeps the TensorFlow E2E path independent of system-wide oneDNN installation while the prototype still uses oneDNN `3.1.1` via `dnnl::matmul` for QK/PV tiles.
## oneDNN tile primitive cache

The oneDNN tile matmul path now keeps a process-local cache keyed by `(M,N,K)`. The cache reuses the CPU engine, stream, memory descriptors, and `dnnl::matmul` primitive for repeated QK/PV tile shapes. This reduced the standalone `flash_attention_qk_pv_onednn_ms` microbenchmark for `M=N=128,K=D=64` from roughly `0.87ms` to `0.717696ms` in the current local run.

This is still a coarse MVP cache: calls are serialized by a mutex, tile inputs are copied before reaching oneDNN, and TensorFlow Custom Op execution still allocates temporary vectors. It validates the primitive reuse direction but is not the final performance architecture.

