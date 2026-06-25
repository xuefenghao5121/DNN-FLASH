#include "flashone/tile_kernel.hpp"

#include <stdexcept>

namespace flashone {

std::vector<float> matmul_tile_onednn(const std::vector<float>& a,
                                      const std::vector<float>& b,
                                      const MatmulShape& shape);

const char* tile_kernel_name(TileKernelKind kind) {
    switch (kind) {
        case TileKernelKind::Reference:
            return "reference";
        case TileKernelKind::OneDnn:
            return "onednn";
    }
    return "unknown";
}

std::vector<float> matmul_tile_reference(const std::vector<float>& a,
                                         const std::vector<float>& b,
                                         const MatmulShape& shape) {
    if (shape.m == 0 || shape.n == 0 || shape.k == 0) {
        throw std::invalid_argument("matmul dimensions must be non-zero");
    }
    if (a.size() != shape.m * shape.k) {
        throw std::invalid_argument("A size does not match matmul shape");
    }
    if (b.size() != shape.k * shape.n) {
        throw std::invalid_argument("B size does not match matmul shape");
    }

    std::vector<float> c(shape.m * shape.n, 0.0f);
    for (std::size_t i = 0; i < shape.m; ++i) {
        for (std::size_t j = 0; j < shape.n; ++j) {
            float sum = 0.0f;
            for (std::size_t kk = 0; kk < shape.k; ++kk) {
                sum += a[i * shape.k + kk] * b[kk * shape.n + j];
            }
            c[i * shape.n + j] = sum;
        }
    }
    return c;
}

#ifndef FLASHONE_HAS_ONEDNN
std::vector<float> matmul_tile_onednn(const std::vector<float>&,
                                      const std::vector<float>&,
                                      const MatmulShape&) {
    throw std::runtime_error("FlashOne was built without oneDNN support");
}
#endif

std::vector<float> matmul_tile(TileKernelKind kind,
                               const std::vector<float>& a,
                               const std::vector<float>& b,
                               const MatmulShape& shape) {
    switch (kind) {
        case TileKernelKind::Reference:
            return matmul_tile_reference(a, b, shape);
        case TileKernelKind::OneDnn:
            return matmul_tile_onednn(a, b, shape);
    }
    throw std::invalid_argument("Unknown tile kernel kind");
}

}  // namespace flashone
