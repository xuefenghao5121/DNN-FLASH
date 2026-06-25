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

// Row-major C = A[M,K] x B[K,N].
// This is the low-level seam that will later replace QK and PV tile loops.
std::vector<float> matmul_tile(TileKernelKind kind,
                               const std::vector<float>& a,
                               const std::vector<float>& b,
                               const MatmulShape& shape);

std::vector<float> matmul_tile_reference(const std::vector<float>& a,
                                         const std::vector<float>& b,
                                         const MatmulShape& shape);

}  // namespace flashone
