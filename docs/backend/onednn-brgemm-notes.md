# oneDNN / BRGEMM Integration Notes

## Current local environment check

Checked on 2026-06-25:

- C++ compiler: GCC 13.3.0
- CMake: 4.3.1
- TBB headers are present under `/usr/include/oneapi/tbb`
- oneDNN development headers/libraries were **not** found:
  - no `dnnl.hpp`
  - no `dnnl.h`
  - no `libdnnl.so`
  - no `pkg-config dnnl`

So the current repository keeps oneDNN integration behind a backend seam and continues to build without oneDNN. The next machine/environment step is installing or building oneDNN with experimental ukernel/BRGEMM support.

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

## Immediate next coding step

Add a build-time optional `FLASHONE_ENABLE_ONEDNN` flag:

- If `dnnl.hpp` and `libdnnl` are available, compile a oneDNN smoke target.
- If unavailable, skip the target and keep all reference tests green.

This avoids blocking algorithm development on local library availability.
