#pragma once

#include <cstddef>
#include <vector>

namespace flashone {

#ifdef FLASHONE_HAS_ONEDNN_BRGEMM
struct BrgemmKernelContext;
#endif

struct MatmulShape {
    std::size_t m;
    std::size_t n;
    std::size_t k;
};

struct StridedMatmulShape {
    std::size_t m;
    std::size_t n;
    std::size_t k;
    std::size_t a_stride_m;
    std::size_t a_stride_k;
    std::size_t b_stride_k;
    std::size_t b_stride_n;
    std::size_t c_stride_m;
    std::size_t c_stride_n;
};

enum class TileKernelKind {
    Reference,
    OneDnn,
    OneDnnBrgemm,
};

const char* tile_kernel_name(TileKernelKind kind);

// Inplace variants: write directly to caller's buffer, no allocation.
void matmul_tile_inplace(TileKernelKind kind,
                         const float* a, const float* b, float* c,
                         const MatmulShape& shape);

#ifdef FLASHONE_HAS_ONEDNN_BRGEMM
// Variant that lets callers provide BRGEMM scratchpad storage for reuse across
// hot tile iterations. Non-BRGEMM kernels ignore the context.
void matmul_tile_inplace(TileKernelKind kind,
                         const float* a, const float* b, float* c,
                         const MatmulShape& shape,
                         BrgemmKernelContext& brgemm_context);
#endif

// Strided inplace variant for non-contiguous tile views. This avoids materializing
// simple transposes such as K_tile^T when the source tensor is row-major K[N,D].
void matmul_tile_strided_inplace(TileKernelKind kind,
                                 const float* a,
                                 const float* b,
                                 float* c,
                                 const StridedMatmulShape& shape);

// Row-major C = A[M,K] x B[K,N].
// This is the low-level seam that will later replace QK and PV tile loops.
std::vector<float> matmul_tile(TileKernelKind kind,
                               const std::vector<float>& a,
                               const std::vector<float>& b,
                               const MatmulShape& shape);

std::vector<float> matmul_tile_reference(const std::vector<float>& a,
                                         const std::vector<float>& b,
                                         const MatmulShape& shape);

void matmul_tile_reference_inplace(const float* a, const float* b, float* c,
                                    const MatmulShape& shape);

}  // namespace flashone
