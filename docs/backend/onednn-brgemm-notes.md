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
