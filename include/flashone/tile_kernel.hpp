#pragma once

#include <cstddef>
#include <vector>

namespace flashone {

struct MatmulShape {
    std::size_t m;
    std::size_t n;
    std::size_t k;
};

enum class TileKernelKind {
    Reference,
    OneDnn,
};

const char* tile_kernel_name(TileKernelKind kind);

// Inplace variants: write directly to caller's buffer, no allocation.
void matmul_tile_inplace(TileKernelKind kind,
                         const float* a, const float* b, float* c,
                         const MatmulShape& shape);

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
