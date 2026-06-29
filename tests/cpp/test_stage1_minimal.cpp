/**
 * Stage 1 Minimal Validation: oneDNN scale post-op
 *
 * Purpose: Verify that oneDNN matmul can handle scale post-op
 * Method: Create new primitive each time (no cache), verify correctness
 * Scope: C++ functional validation only, no performance claims
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include <oneapi/dnnl/dnnl.hpp>

namespace {

constexpr float kScale = 0.125f;  // 1 / sqrt(64)
constexpr float kTolerance = 1e-5f;

// Reference implementation: Q * K^T * scale
void reference_qk_scale(const float* q, const float* k, float* score,
                        int m, int n, int k_dim, float scale) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k_dim; ++l) {
                sum += q[i * k_dim + l] * k[j * k_dim + l];
            }
            score[i * n + j] = sum * scale;
        }
    }
}

// Helper to create memory descriptor (from existing codebase)
dnnl::memory::desc md2(std::size_t dim0, std::size_t dim1,
                       std::size_t stride0, std::size_t stride1) {
    return dnnl::memory::desc({static_cast<dnnl::memory::dim>(dim0),
                               static_cast<dnnl::memory::dim>(dim1)},
                              dnnl::memory::data_type::f32,
                              {static_cast<dnnl::memory::dim>(stride0),
                               static_cast<dnnl::memory::dim>(stride1)});
}

// oneDNN implementation with scale post-op
void onednn_qk_scale(const float* q, const float* k, float* score,
                     int m, int n, int k_dim, float scale) {
    using namespace dnnl;

    // Create engine and stream (new each time, no cache)
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    // Create memory descriptors (contiguous row-major: strides are k_dim and n)
    auto q_md = md2(m, k_dim, k_dim, 1);      // [m, k] row-major
    auto k_md = md2(n, k_dim, k_dim, 1);      // [n, k] row-major
    auto score_md = md2(m, n, n, 1);          // [m, n] row-major

    // Create memory objects
    memory q_mem(q_md, eng, const_cast<float*>(q));
    memory k_mem(k_md, eng, const_cast<float*>(k));
    memory score_mem(score_md, eng, score);

    // Create matmul primitive descriptor
    matmul::primitive_desc matmul_pd(eng, q_md, k_md, score_md);

    // Add scale post-op
    post_ops po;
    po.append_eltwise(algorithm::eltwise_linear, scale, 0.0f);

    primitive_attr attr;
    attr.set_post_ops(po);

    // Create primitive descriptor with post-ops
    matmul::primitive_desc matmul_po_pd(eng, q_md, k_md, score_md, attr);
    matmul matmul_prim(matmul_po_pd);

    // Execute
    matmul_prim.execute(s, {
        {DNNL_ARG_SRC_0, q_mem},
        {DNNL_ARG_SRC_1, k_mem},
        {DNNL_ARG_DST, score_mem}
    });
    s.wait();
}

// Test case
bool test_scale_postop(int m, int n, int k_dim) {
    // Allocate test data
    std::vector<float> q(m * k_dim);
    std::vector<float> k(n * k_dim);
    std::vector<float> score_ref(m * n);
    std::vector<float> score_onednn(m * n);

    // Initialize with test data
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = static_cast<float>(i) * 0.1f;
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = static_cast<float>(i + 100) * 0.1f;
    }

    // Reference implementation
    reference_qk_scale(q.data(), k.data(), score_ref.data(),
                       m, n, k_dim, kScale);

    // oneDNN implementation
    onednn_qk_scale(q.data(), k.data(), score_onednn.data(),
                   m, n, k_dim, kScale);

    // Compare results
    float max_diff = 0.0f;
    for (int i = 0; i < m * n; ++i) {
        float diff = std::abs(score_ref[i] - score_onednn[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > kTolerance) {
            std::cerr << "Mismatch at (" << i / n << ", " << i % n << "): "
                      << "ref=" << score_ref[i] << ", onednn=" << score_onednn[i]
                      << ", diff=" << diff << std::endl;
            return false;
        }
    }

    std::cout << "  PASSED: max_diff=" << max_diff << std::endl;
    return true;
}

}  // namespace

int main() {
    std::cout << "Stage 1 Minimal Validation: oneDNN scale post-op" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "oneDNN version: " << dnnl::version() << std::endl;

    bool all_passed = true;

    // Test different tile sizes
    std::cout << "\nTest 1: 64x64 tile, head_dim=64" << std::endl;
    all_passed &= test_scale_postop(64, 64, 64);

    std::cout << "\nTest 2: 32x128 tile, head_dim=64" << std::endl;
    all_passed &= test_scale_postop(32, 128, 64);

    std::cout << "\nTest 3: 128x32 tile, head_dim=64" << std::endl;
    all_passed &= test_scale_postop(128, 32, 64);

    std::cout << "\nTest 4: Small tile 16x16, head_dim=32" << std::endl;
    all_passed &= test_scale_postop(16, 16, 32);

    std::cout << "\n======================================================" << std::endl;
    if (all_passed) {
        std::cout << "All tests PASSED" << std::endl;
        std::cout << "\nConclusion: oneDNN matmul supports scale post-op" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED" << std::endl;
        return 1;
    }
}
