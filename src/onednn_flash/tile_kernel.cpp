#include "onednn_flash/tile_kernel.hpp"

#ifdef ONEDNN_FLASH_HAS_ONEDNN_BRGEMM
#include "onednn_flash/onednn_brgemm_tile_kernel.hpp"
#endif

#include <stdexcept>

namespace onednn_flash {

std::vector<float> matmul_tile_onednn(const std::vector<float>& a,
                                      const std::vector<float>& b,
                                      const MatmulShape& shape);
void matmul_tile_onednn_inplace(const float* a, const float* b, float* c,
                                 const MatmulShape& shape);
void matmul_tile_onednn_strided_inplace(const float* a, const float* b, float* c,
                                        const StridedMatmulShape& shape);

const char* tile_kernel_name(TileKernelKind kind) {
    switch (kind) {
        case TileKernelKind::Reference:
            return "reference";
        case TileKernelKind::OneDnn:
            return "onednn";
        case TileKernelKind::OneDnnBrgemm:
            return "onednn_brgemm";
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
    matmul_tile_reference_inplace(a.data(), b.data(), c.data(), shape);
    return c;
}

void matmul_tile_reference_inplace(const float* a, const float* b, float* c,
                                   const MatmulShape& shape) {
    if (shape.m == 0 || shape.n == 0 || shape.k == 0) {
        throw std::invalid_argument("matmul dimensions must be non-zero");
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument("matmul pointers must be non-null");
    }
    for (std::size_t i = 0; i < shape.m; ++i) {
        for (std::size_t j = 0; j < shape.n; ++j) {
            float sum = 0.0f;
            for (std::size_t kk = 0; kk < shape.k; ++kk) {
                sum += a[i * shape.k + kk] * b[kk * shape.n + j];
            }
            c[i * shape.n + j] = sum;
        }
    }
}

void matmul_tile_reference_strided_inplace(const float* a, const float* b, float* c,
                                           const StridedMatmulShape& shape) {
    if (shape.m == 0 || shape.n == 0 || shape.k == 0) {
        throw std::invalid_argument("matmul dimensions must be non-zero");
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument("matmul pointers must be non-null");
    }
    for (std::size_t i = 0; i < shape.m; ++i) {
        for (std::size_t j = 0; j < shape.n; ++j) {
            float sum = 0.0f;
            for (std::size_t kk = 0; kk < shape.k; ++kk) {
                sum += a[i * shape.a_stride_m + kk * shape.a_stride_k] *
                       b[kk * shape.b_stride_k + j * shape.b_stride_n];
            }
            c[i * shape.c_stride_m + j * shape.c_stride_n] = sum;
        }
    }
}

#ifndef ONEDNN_FLASH_HAS_ONEDNN
std::vector<float> matmul_tile_onednn(const std::vector<float>&,
                                      const std::vector<float>&,
                                      const MatmulShape&) {
    throw std::runtime_error("OneDNNFlash was built without oneDNN support");
}
void matmul_tile_onednn_inplace(const float*, const float*, float*,
                                 const MatmulShape&) {
    throw std::runtime_error("OneDNNFlash was built without oneDNN support");
}
void matmul_tile_onednn_strided_inplace(const float*, const float*, float*,
                                        const StridedMatmulShape&) {
    throw std::runtime_error("OneDNNFlash was built without oneDNN support");
}
#endif

#ifndef ONEDNN_FLASH_HAS_ONEDNN_BRGEMM
void matmul_tile_onednn_brgemm_inplace(const float*, const float*, float*,
                                       const MatmulShape&) {
    throw std::runtime_error("OneDNNFlash was built without oneDNN BRGEMM ukernel support");
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
        case TileKernelKind::OneDnnBrgemm: {
            std::vector<float> c(shape.m * shape.n, 0.0f);
            matmul_tile_onednn_brgemm_inplace(a.data(), b.data(), c.data(), shape);
            return c;
        }
    }
    throw std::invalid_argument("Unknown tile kernel kind");
}

void matmul_tile_inplace(TileKernelKind kind, const float* a, const float* b, float* c,
                          const MatmulShape& shape) {
    switch (kind) {
        case TileKernelKind::Reference:
            matmul_tile_reference_inplace(a, b, c, shape);
            return;
        case TileKernelKind::OneDnn:
            matmul_tile_onednn_inplace(a, b, c, shape);
            return;
        case TileKernelKind::OneDnnBrgemm:
            matmul_tile_onednn_brgemm_inplace(a, b, c, shape);
            return;
    }
    throw std::invalid_argument("Unknown tile kernel kind");
}

#ifdef ONEDNN_FLASH_HAS_ONEDNN_BRGEMM
void matmul_tile_inplace(TileKernelKind kind,
                         const float* a,
                         const float* b,
                         float* c,
                         const MatmulShape& shape,
                         BrgemmKernelContext& brgemm_context) {
    switch (kind) {
        case TileKernelKind::Reference:
            matmul_tile_reference_inplace(a, b, c, shape);
            return;
        case TileKernelKind::OneDnn:
            matmul_tile_onednn_inplace(a, b, c, shape);
            return;
        case TileKernelKind::OneDnnBrgemm:
            matmul_tile_onednn_brgemm_inplace(
                a, b, c, shape, brgemm_context.scratchpad, false);
            brgemm_context.hw_context_active = true;
            return;
    }
    throw std::invalid_argument("Unknown tile kernel kind");
}
#endif

void matmul_tile_strided_inplace(TileKernelKind kind,
                                 const float* a,
                                 const float* b,
                                 float* c,
                                 const StridedMatmulShape& shape) {
    switch (kind) {
        case TileKernelKind::Reference:
            matmul_tile_reference_strided_inplace(a, b, c, shape);
            return;
        case TileKernelKind::OneDnn:
            matmul_tile_onednn_strided_inplace(a, b, c, shape);
            return;
        case TileKernelKind::OneDnnBrgemm:
            throw std::runtime_error("oneDNN BRGEMM tile kernel requires contiguous A/B/C tiles; strided path is not implemented yet");
    }
    throw std::invalid_argument("Unknown tile kernel kind");
}

}  // namespace onednn_flash
