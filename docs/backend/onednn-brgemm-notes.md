# oneDNN / BRGEMM Integration Notes

## Current local environment check

Checked on 2026-06-25:

- C++ compiler: GCC 13.3.0
- CMake: 4.3.1
- TBB headers are present under `/usr/include/oneapi/tbb`
- System oneDNN development headers/libraries were not installed, but Ubuntu packages were downloaded and extracted locally under `third_party/onednn-local`. The optional CMake path now builds `flashone_dnnl_smoke` and verifies oneDNN matmul at runtime.
- A newer Debian oneDNN 3.12.1 package with experimental ukernel headers is also available under `third_party/onednn-debian-3.12`. CMake now searches this path before `third_party/onednn-local`, and enables the BRGEMM ukernel tile path when `<oneapi/dnnl/dnnl_ukernel.hpp>` plus `dnnl::ukernel::brgemm` compile successfully.

The repository still keeps oneDNN integration behind a backend seam so the reference path continues to build when oneDNN is unavailable.

## Why a backend seam now

The standalone tiled attention core already exposes the algorithmic replacement point:

```text
run_attention(kind, q, k, v, shape, options)
```

Current backend kinds:

- `StandardReference`
- `FlashTiledReference`
- `OneDnnBrgemmReferenceLayout` (contiguous f32 QK/PV tile path; experimental ukernel)

Future backend kinds:

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

## First real oneDNN BRGEMM ukernel path — 2026-06-27

FlashOne now has an actual oneDNN experimental BRGEMM ukernel backend behind the same `TileKernel` seam:

```cpp
enum class TileKernelKind {
    Reference,
    OneDnn,
    OneDnnBrgemm,
};
```

Build detection:

- CMake checks `<oneapi/dnnl/dnnl_ukernel.hpp>` and `dnnl::ukernel::brgemm::get_B_pack_type(...)` with `DNNL_EXPERIMENTAL_UKERNEL`.
- If the check succeeds, `src/flashone/onednn_brgemm_tile_kernel.cpp` is compiled and `FLASHONE_HAS_ONEDNN_BRGEMM=1` is exported.
- The local working build used `third_party/onednn-debian-3.12/usr` because the older `third_party/onednn-local` package does not expose the ukernel C++ header.

CMake command used for the verified BRGEMM build:

```bash
cmake -S . -B build-brgemm-main \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLASHONE_BUILD_TENSORFLOW_OP=ON \
  -DFLASHONE_ENABLE_ONEDNN=ON \
  -DDNNL_INCLUDE_DIR=$PWD/third_party/onednn-debian-3.12/usr/include \
  -DDNNL_LIBRARY=$PWD/third_party/onednn-debian-3.12/usr/lib/x86_64-linux-gnu/libdnnl.so
cmake --build build-brgemm-main -j2
```

Scope of this first integration:

- f32 only.
- Contiguous A/B/C tile only.
- `qk_tile_layout='copied_transposed'` required for TensorFlow Custom Op `tile_kernel='onednn_brgemm'`.
- Strided-K BRGEMM is intentionally rejected until an offset/pack implementation is added.
- `BrgemmShape` supports batch-reduce semantics for `C[M,N] = sum_i A_i[M,K] x B_i[K,N]`, and has a C++ unit test.
- Current attention path still calls BRGEMM as individual QK/PV tile matmuls (`batch_size=1`); using batch-reduce to fuse PV accumulation is a later step.

Validation on local x86 host:

```text
ctest --test-dir build-brgemm-main --output-on-failure
100% tests passed, 0 tests failed out of 7

TensorFlow custom op smoke:
tile_kernel='onednn_brgemm', qk_tile_layout='copied_transposed'
tf_brgemm_shape (1, 2, 32, 16)
tf_brgemm_max_diff_vs_onednn 0.0
```

Microbenchmark (`M=N=128,K=D=64,causal=1,query_block=16,key_block=32`):

```text
max_abs_diff_qk_pv_onednn_brgemm: 1.86265e-08
flash_attention_qk_pv_onednn_ms: 0.153394
flash_attention_qk_pv_onednn_copied_k_ms: 0.151643
flash_attention_qk_pv_onednn_brgemm_ms: 0.101595
```

Interpretation:

- The project no longer only uses `dnnl::matmul`; there is now a real `dnnl::ukernel::brgemm` path.
- At this small reference shape, BRGEMM is about 1.5x faster than the current oneDNN matmul tile path.
- This still does not prove the final FlashAttention design claim; online softmax remains in FlashOne C++ and BRGEMM is not yet doing packed-B, bf16/AMX, post-op softmax, or multi-tile batch-reduce PV accumulation.

Next BRGEMM-specific steps:

1. Add explicit B-pack/transform support if `get_B_pack_type(f32,f32)` requires packing on another host/ISA.
2. Add a strided/offset BRGEMM path for row-major K to avoid copied-transposed K when using BRGEMM.
3. Use `BrgemmShape.batch_size > 1` where it actually reduces multiple products, rather than only using the ukernel as a single GEMM replacement.
4. Re-run TF E2E benchmark with `tile_kernel='onednn_brgemm'` and compare graph/eager overhead separately.

## BRGEMM scratchpad reuse — 2026-06-27

The first real BRGEMM path no longer allocates a scratchpad vector on every tile call. The low-level API now exposes reusable storage:

```cpp
struct BrgemmScratchpad {
    std::vector<std::uint8_t> bytes;

    void* data(std::size_t required_size);
};

void matmul_tile_onednn_brgemm_inplace(..., BrgemmScratchpad& scratchpad);
```

`AttentionWorkspace` owns one `brgemm_scratchpad`, and the QK/PV workspace path passes it into `matmul_tile_inplace(...)` when `FLASHONE_HAS_ONEDNN_BRGEMM` is enabled. The older no-scratchpad overloads remain available and use `thread_local BrgemmScratchpad`, so tests and one-off calls still avoid repeated heap allocation after the first resize. `BrgemmScratchpad::data(...)` returns a 64-byte aligned pointer because oneDNN's ukernel implementation checks cache-line alignment for scratchpad.

The same step also starts using BRGEMM's batch-reduce semantics in the generic contiguous `MatmulShape` path: when `K` is divisible by 16, FlashOne feeds the ukernel `batch_size=K/16` with `k=16` slices instead of issuing a single `batch_size=1` GEMM replacement. For the current attention microbenchmark this means QK with `D=64` uses four reduced products, and PV with `Kb=32` uses two reduced products.

Validation after this change:

```text
cmake --build build-brgemm-main -j2
ctest --test-dir build-brgemm-main --output-on-failure
100% tests passed, 0 tests failed out of 7

PYTHONPATH=python:. python3 -m pytest -q tests/python tests/tensorflow
16 passed

ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 7

TensorFlow custom op smoke:
tile_kernel='onednn_brgemm', qk_tile_layout='copied_transposed'
tf_brgemm_shape (1, 2, 32, 16)
tf_brgemm_max_diff_vs_onednn 0.0
```

The same step also avoids releasing oneDNN's BRGEMM hardware context after every tile in the workspace attention path. `BrgemmKernelContext` tracks whether a BRGEMM call initialized the per-thread hardware state; `flash_attention_qk_pv_tile_ws(...)` now releases it once at the end of the workspace call, while the legacy one-off overloads still release immediately.

Microbenchmark after scratchpad reuse + K-split batch-reduce + delayed hw-context release (`M=N=128,K=D=64,causal=1,query_block=16,key_block=32`):

```text
max_abs_diff_qk_pv_onednn_brgemm: 1.86265e-08
flash_attention_qk_pv_onednn_ms: 0.144895
flash_attention_qk_pv_onednn_copied_k_ms: 0.143994
flash_attention_qk_pv_onednn_brgemm_ms: 0.0947744
```

Earlier microbenchmark after only scratchpad reuse + K-split batch-reduce:

```text
max_abs_diff_qk_pv_onednn_brgemm: 1.86265e-08
flash_attention_qk_pv_onednn_ms: 0.150179
flash_attention_qk_pv_onednn_copied_k_ms: 0.149363
flash_attention_qk_pv_onednn_brgemm_ms: 0.0996274
```

Interpretation:

- The hot QK/PV BRGEMM path is now free of per-tile scratchpad allocation.
- The scratchpad passed to oneDNN is cache-line aligned instead of relying on `std::vector<uint8_t>`'s default alignment.
- The contiguous `MatmulShape` BRGEMM route now uses real batch-reduce across K slices for K values divisible by 16, so it is no longer merely a `batch_size=1` single-GEMM wrapper in the common FlashOne attention shapes.
- The next high-value step is a row-major-K BRGEMM path, because copied-transposed K is now the bigger remaining design compromise.

After delayed hw-context release, the measured BRGEMM tile path moved from `~0.0996ms` to `~0.0948ms` in the same local microbenchmark. Treat that as directional rather than final benchmarking, but it confirms the per-tile release was measurable overhead.

## BRGEMM transformed-K QK path — 2026-06-27

A new experimental QK layout is available:

```text
qk_tile_layout='brgemm_transformed_k'
```

This path targets the remaining compromise in `tile_kernel='onednn_brgemm'`: the previous BRGEMM QK route required FlashOne to explicitly copy/transpose each K block into `[D,Kb]` before calling BRGEMM. Directly feeding the original row-major K block `[Kb,D]` as a strided transposed B view is not expressible with the current oneDNN BRGEMM ukernel API: it exposes leading dimensions and batch offsets for contiguous operands, but not an arbitrary per-element stride pattern for `B=[D,Kb]` over source `K=[Kb,D]`.

Implementation:

- `QkTileLayout::BrgemmTransformedK` was added to C++ attention options.
- TensorFlow/Python accepts `qk_tile_layout='brgemm_transformed_k'` when `tile_kernel='onednn_brgemm'`.
- `matmul_tile_onednn_brgemm_transposed_b_inplace(...)` takes source `b_transposed=[N,K]` and materializes BRGEMM-consumable `B=[K,N]` into `BrgemmTransformWorkspace`.
- For transform output leading dimensions supported by oneDNN's ukernel transform docs (`out_ld` in `16/32/48/64`), it uses `dnnl::ukernel::transform` with `pack_type::trans`.
- For unsupported small/partial `N`, it falls back to a tiny reference materialization so the path remains correct for tests and edge tiles.
- `AttentionWorkspace` owns the transform workspace via `BrgemmKernelContext`, so the transformed-B buffer is reused across hot QK tiles instead of allocated per tile.

Validation after adding this path:

```text
cmake --build build-brgemm-main -j2
ctest --test-dir build-brgemm-main --output-on-failure
100% tests passed, 0 tests failed out of 7

PYTHONPATH=$PWD/python:$PWD python3 -m pytest -q tests/python tests/tensorflow
17 passed
```

The TensorFlow test suite includes an isolated subprocess smoke for `tile_kernel='onednn_brgemm'` with both `qk_tile_layout='copied_transposed'` and `qk_tile_layout='brgemm_transformed_k'`, because TensorFlow cannot safely load two shared libraries registering the same `FlashOneAttention` op in one process.

Latest local C++ benchmark (`M=N=128,K=D=64,causal=1,query_block=16,key_block=32`) after transformed-K support:

```text
max_abs_diff_qk_pv_onednn_brgemm: 1.86265e-08
max_abs_diff_qk_pv_onednn_brgemm_transformed_k: 1.86265e-08
flash_attention_qk_pv_onednn_ms: 0.268732
flash_attention_qk_pv_onednn_copied_k_ms: 0.273044
flash_attention_qk_pv_onednn_brgemm_ms: 0.166301
flash_attention_qk_pv_onednn_brgemm_transformed_k_ms: 0.151441
```

Interpretation:

- This does not make BRGEMM consume row-major K with zero materialization; oneDNN BRGEMM still needs contiguous/packed B.
- It moves the K transpose/copy responsibility from ad-hoc nested loops to a reusable transform path and uses oneDNN's transform ukernel where the documented output leading dimension permits it.
- On the current benchmark shape, transformed-K is slightly faster than the copied-transposed BRGEMM path, but treat this as an experimental path until broader shape sweeps are collected.
- Python default remains `qk_tile_layout='strided_k'` for the non-BRGEMM TensorFlow path because earlier TF eager benchmarks showed that avoiding K materialization is consistently helpful there.

Remaining BRGEMM work:

1. Broaden transformed-K benchmark sweeps across `seq/head_dim/value_dim/tile` and separate edge tiles where `N` is not 16/32/48/64.
2. Investigate whether oneDNN's lower-level C ukernel APIs expose a more direct offset-array or pack path for row-major K without temporary materialization.
3. Add a transform/pack path for platforms where `brgemm::get_B_pack_type(f32,f32)` is not `pack_type::no_trans`; the current BRGEMM constructor still rejects that case.
4. Continue toward TensorFlow graph/XLA integration, because eager custom-op overhead remains outside the tile microbenchmark.

## oneDNN tile primitive cache

The oneDNN tile matmul path now keeps a process-local cache keyed by `(M,N,K)`. The cache reuses the CPU engine, stream, memory descriptors, and `dnnl::matmul` primitive for repeated QK/PV tile shapes. This reduced the standalone `flash_attention_qk_pv_onednn_ms` microbenchmark for `M=N=128,K=D=64` from roughly `0.87ms` to `0.717696ms` in the current local run.

This is still a coarse MVP cache: calls are serialized by a mutex, tile inputs are copied before reaching oneDNN, and TensorFlow Custom Op execution still allocates temporary vectors. It validates the primitive reuse direction but is not the final performance architecture.

