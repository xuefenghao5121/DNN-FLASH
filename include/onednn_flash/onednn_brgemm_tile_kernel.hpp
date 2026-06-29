#pragma once

#include "onednn_flash/tile_kernel.hpp"

#include <cstdint>
#include <vector>

namespace onednn_flash {

// Reusable BRGEMM scratchpad. oneDNN ukernel exposes scratchpad size per
// generated kernel; keeping the backing storage outside the hot call avoids a
// heap allocation per QK/PV tile. oneDNN's ukernel implementation expects
// cache-line-aligned scratchpad for best behavior, so data() returns a 64-byte
// aligned pointer into the backing buffer.
struct BrgemmScratchpad {
    std::vector<std::uint8_t> bytes;

    void* data(std::size_t required_size);
};

struct BrgemmTransformWorkspace {
    std::vector<std::uint8_t> bytes;

    void* data(std::size_t required_size);
};

struct BrgemmKernelContext {
    BrgemmScratchpad scratchpad;
    BrgemmTransformWorkspace transform_workspace;
    // Set after a BRGEMM ukernel has initialized oneDNN's per-thread hardware
    // context. Attention code releases it at the end of a tiled workspace call;
    // the destructor is a safety net for one-off/thread-local users.
    bool hw_context_active = false;

    ~BrgemmKernelContext();
};

// BRGEMM shape for C[M,N] = sum_i A_i[M,K] x B_i[K,N].
// Leading dimensions and batch strides are in elements, not bytes.
struct BrgemmShape {
    std::size_t m;
    std::size_t n;
    std::size_t k;
    std::size_t batch_size;
    std::size_t lda;
    std::size_t ldb;
    std::size_t ldc;
    std::size_t a_batch_stride;
    std::size_t b_batch_stride;
};

bool onednn_brgemm_available();

void release_onednn_brgemm_hw_context();

void matmul_tile_onednn_brgemm_inplace(const float* a,
                                       const float* b,
                                       float* c,
                                       const MatmulShape& shape);

void matmul_tile_onednn_brgemm_inplace(const float* a,
                                       const float* b,
                                       float* c,
                                       const MatmulShape& shape,
                                       BrgemmScratchpad& scratchpad);

void matmul_tile_onednn_brgemm_inplace(const float* a,
                                       const float* b,
                                       float* c,
                                       const MatmulShape& shape,
                                       BrgemmScratchpad& scratchpad,
                                       bool release_hw_context);

void matmul_tile_onednn_brgemm_inplace(const float* a,
                                       const float* b,
                                       float* c,
                                       const BrgemmShape& shape);

void matmul_tile_onednn_brgemm_inplace(const float* a,
                                       const float* b,
                                       float* c,
                                       const BrgemmShape& shape,
                                       BrgemmScratchpad& scratchpad);

void matmul_tile_onednn_brgemm_inplace(const float* a,
                                       const float* b,
                                       float* c,
                                       const BrgemmShape& shape,
                                       BrgemmScratchpad& scratchpad,
                                       bool release_hw_context);

void matmul_tile_onednn_brgemm_transposed_b_inplace(const float* a,
                                                    const float* b_transposed,
                                                    float* c,
                                                    const MatmulShape& shape,
                                                    BrgemmScratchpad& scratchpad,
                                                    BrgemmTransformWorkspace& transform_workspace,
                                                    bool release_hw_context);

void matmul_tile_onednn_brgemm_transposed_b_inplace(const float* a,
                                                    const float* b_transposed,
                                                    float* c,
                                                    const MatmulShape& shape,
                                                    BrgemmKernelContext& context,
                                                    bool release_hw_context);

}  // namespace onednn_flash
