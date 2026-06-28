#include "flashone/tile_kernel.hpp"

#ifdef FLASHONE_HAS_ONEDNN_BRGEMM
#include "flashone/onednn_brgemm_tile_kernel.hpp"
#endif

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> make_values(std::size_t size, float offset) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.19f + offset) * 0.4f;
    }
    return values;
}

void require_close(const std::vector<float>& a,
                   const std::vector<float>& b,
                   float tolerance,
                   const char* message) {
    if (a.size() != b.size()) {
        throw std::runtime_error("size mismatch");
    }
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    if (max_diff > tolerance) {
        std::cerr << message << " max_diff=" << max_diff << "\n";
        throw std::runtime_error(message);
    }
}

void test_reference_dispatch() {
    const flashone::MatmulShape shape{5, 7, 3};
    const auto a = make_values(shape.m * shape.k, 0.2f);
    const auto b = make_values(shape.k * shape.n, 1.2f);
    const auto direct = flashone::matmul_tile_reference(a, b, shape);
    const auto dispatched = flashone::matmul_tile(flashone::TileKernelKind::Reference, a, b, shape);
    require_close(direct, dispatched, 1e-6f, "reference dispatch mismatch");
}

#ifdef FLASHONE_HAS_ONEDNN
void test_onednn_matches_reference() {
    const flashone::MatmulShape shape{8, 9, 6};
    const auto a = make_values(shape.m * shape.k, 0.4f);
    const auto b = make_values(shape.k * shape.n, 1.4f);
    const auto expected = flashone::matmul_tile_reference(a, b, shape);
    const auto actual = flashone::matmul_tile(flashone::TileKernelKind::OneDnn, a, b, shape);
    require_close(expected, actual, 1e-5f, "oneDNN tile kernel mismatch");
}
#endif

#ifdef FLASHONE_HAS_ONEDNN_BRGEMM
void test_onednn_brgemm_matches_reference() {
    const flashone::MatmulShape shape{16, 16, 16};
    const auto a = make_values(shape.m * shape.k, 0.7f);
    const auto b = make_values(shape.k * shape.n, 1.7f);
    const auto expected = flashone::matmul_tile_reference(a, b, shape);
    const auto actual = flashone::matmul_tile(flashone::TileKernelKind::OneDnnBrgemm, a, b, shape);
    require_close(expected, actual, 1e-4f, "oneDNN BRGEMM tile kernel mismatch");
}

void test_onednn_brgemm_k_split_matmul_matches_reference() {
    const flashone::MatmulShape shape{13, 11, 32};
    const auto a = make_values(shape.m * shape.k, 0.8f);
    const auto b = make_values(shape.k * shape.n, 1.8f);
    const auto expected = flashone::matmul_tile_reference(a, b, shape);
    const auto actual = flashone::matmul_tile(flashone::TileKernelKind::OneDnnBrgemm, a, b, shape);
    require_close(expected, actual, 1e-4f, "oneDNN BRGEMM K-split matmul mismatch");
}

void test_onednn_brgemm_batch_reduce() {
    const std::size_t batch = 3;
    const flashone::BrgemmShape shape{/*m=*/8,
                                      /*n=*/8,
                                      /*k=*/4,
                                      batch,
                                      /*lda=*/4,
                                      /*ldb=*/8,
                                      /*ldc=*/8,
                                      /*a_batch_stride=*/8 * 4,
                                      /*b_batch_stride=*/4 * 8};
    const auto a = make_values(batch * shape.a_batch_stride, 0.9f);
    const auto b = make_values(batch * shape.b_batch_stride, 1.9f);
    std::vector<float> actual(shape.m * shape.n, 0.0f);
    std::vector<float> expected(shape.m * shape.n, 0.0f);

    flashone::matmul_tile_onednn_brgemm_inplace(a.data(), b.data(), actual.data(), shape);

    for (std::size_t bb = 0; bb < batch; ++bb) {
        for (std::size_t i = 0; i < shape.m; ++i) {
            for (std::size_t j = 0; j < shape.n; ++j) {
                for (std::size_t kk = 0; kk < shape.k; ++kk) {
                    expected[i * shape.n + j] +=
                        a[bb * shape.a_batch_stride + i * shape.lda + kk] *
                        b[bb * shape.b_batch_stride + kk * shape.ldb + j];
                }
            }
        }
    }
    require_close(expected, actual, 1e-4f, "oneDNN BRGEMM batch-reduce mismatch");
}

void test_onednn_brgemm_scratchpad_alignment() {
    flashone::BrgemmScratchpad scratchpad;
    void* ptr = scratchpad.data(137);
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    if (ptr == nullptr || address % 64 != 0) {
        throw std::runtime_error("oneDNN BRGEMM scratchpad is not 64-byte aligned");
    }
}

void test_onednn_brgemm_transposed_b_matches_reference() {
    const flashone::MatmulShape shape{16, 16, 16};
    const auto a = make_values(shape.m * shape.k, 1.1f);
    const auto b = make_values(shape.k * shape.n, 2.1f);
    std::vector<float> b_transposed(shape.n * shape.k);
    for (std::size_t k = 0; k < shape.k; ++k) {
        for (std::size_t n = 0; n < shape.n; ++n) {
            b_transposed[n * shape.k + k] = b[k * shape.n + n];
        }
    }
    const auto expected = flashone::matmul_tile_reference(a, b, shape);
    std::vector<float> actual(shape.m * shape.n, 0.0f);
    flashone::BrgemmScratchpad scratchpad;
    flashone::BrgemmTransformWorkspace transform_workspace;
    flashone::matmul_tile_onednn_brgemm_transposed_b_inplace(a.data(),
                                                             b_transposed.data(),
                                                             actual.data(),
                                                             shape,
                                                             scratchpad,
                                                             transform_workspace,
                                                             true);
    require_close(expected, actual, 1e-4f, "oneDNN BRGEMM transposed-B transform mismatch");
}

void test_onednn_brgemm_transform_alignment() {
    flashone::BrgemmTransformWorkspace workspace;
    void* ptr = workspace.data(257);
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    if (ptr == nullptr || address % 64 != 0) {
        throw std::runtime_error("oneDNN BRGEMM transform workspace is not 64-byte aligned");
    }
}
#endif

}  // namespace

int main() {
    test_reference_dispatch();
#ifdef FLASHONE_HAS_ONEDNN
    test_onednn_matches_reference();
#endif
#ifdef FLASHONE_HAS_ONEDNN_BRGEMM
    test_onednn_brgemm_matches_reference();
    test_onednn_brgemm_k_split_matmul_matches_reference();
    test_onednn_brgemm_batch_reduce();
    test_onednn_brgemm_scratchpad_alignment();
    test_onednn_brgemm_transposed_b_matches_reference();
    test_onednn_brgemm_transform_alignment();
#endif
    std::cout << "flashone tile kernel tests passed\n";
    return 0;
}
