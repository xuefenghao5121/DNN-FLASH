#include "flashone/qk_score_tile_internal.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> make_values(std::size_t size, float offset) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.13f + offset) * 0.5f;
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

flashone::RuntimePlanInput base_input() {
    flashone::RuntimePlanInput input;
    input.query_length = 4;
    input.key_length = 5;
    input.head_dim = 3;
    input.value_dim = 3;
    input.enable_onednn = true;
    return input;
}

std::vector<float> reference_scores(const std::vector<float>& q,
                                    const std::vector<float>& k,
                                    const float* bias,
                                    std::size_t bias_stride_m,
                                    std::size_t bias_stride_n,
                                    float scale,
                                    bool use_scale,
                                    std::size_t m,
                                    std::size_t n,
                                    std::size_t d) {
    std::vector<float> out(m * n, 0.0f);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (std::size_t kk = 0; kk < d; ++kk) {
                sum += q[i * d + kk] * k[j * d + kk];
            }
            if (use_scale) {
                sum *= scale;
            }
            if (bias != nullptr) {
                sum += bias[i * bias_stride_m + j * bias_stride_n];
            }
            out[i * n + j] = sum;
        }
    }
    return out;
}

void run_qk_score_tile_case(flashone::ScoreModKind kind,
                            bool has_scale,
                            float scale,
                            bool has_bias) {
    constexpr std::size_t m = 4;
    constexpr std::size_t n = 5;
    constexpr std::size_t d = 3;
    const auto q = make_values(m * d, 0.1f);
    const auto k = make_values(n * d, 0.7f);
    auto bias = make_values(m * n, 1.3f);
    std::vector<float> actual(m * n, 0.0f);

    auto input = base_input();
    input.requested_score_mod = kind;
    input.has_scale = has_scale;
    input.scale_value = scale;
    input.requested_bias_kind = has_bias ? flashone::BiasKind::SameShapeTile : flashone::BiasKind::None;
    const auto plan = flashone::make_runtime_plan(input,
                                                  /*one_dnn_available=*/true,
                                                  /*one_dnn_post_ops_available=*/true);

    flashone::QkScoreTilePostOpsInput post_ops;
    if (has_bias) {
        post_ops.additive_bias = bias.data();
        post_ops.additive_bias_stride_m = n;
        post_ops.additive_bias_stride_n = 1;
    }

    const flashone::StridedMatmulShape shape{/*m=*/m,
                                             /*n=*/n,
                                             /*k=*/d,
                                             /*a_stride_m=*/d,
                                             /*a_stride_k=*/1,
                                             /*b_stride_k=*/1,
                                             /*b_stride_n=*/d,
                                             /*c_stride_m=*/n,
                                             /*c_stride_n=*/1};
    flashone::QkScoreTileDebugInfo debug;
    flashone::qk_score_tile_inplace(q.data(), k.data(), actual.data(), shape, plan, post_ops, &debug);

    const auto expected = reference_scores(q, k, has_bias ? bias.data() : nullptr, n, 1, scale, has_scale, m, n, d);
    require_close(expected, actual, 1e-5f, "QK score tile post-op mismatch");
#ifdef FLASHONE_HAS_ONEDNN
    require(debug.backend == flashone::QkBackendKind::OneDnnMatmul, "expected oneDNN matmul backend");
    require(debug.lowering_status == flashone::LoweringStatus::LoweredToOneDnnPostOps,
            "expected oneDNN post-op lowering");
    require(debug.fallback_reason == flashone::FallbackReason::None, "unexpected fallback reason");
#endif
}

void test_scale_post_op() {
    run_qk_score_tile_case(flashone::ScoreModKind::Scale,
                           /*has_scale=*/true,
                           /*scale=*/0.25f,
                           /*has_bias=*/false);
}

void test_scale_additive_bias_post_op() {
    run_qk_score_tile_case(flashone::ScoreModKind::ScaleAdditiveBias,
                           /*has_scale=*/true,
                           /*scale=*/0.5f,
                           /*has_bias=*/true);
}

void test_reference_fallback_for_broadcast_bias() {
    constexpr std::size_t m = 4;
    constexpr std::size_t n = 5;
    constexpr std::size_t d = 3;
    const auto q = make_values(m * d, 0.2f);
    const auto k = make_values(n * d, 0.8f);
    const auto row_bias = make_values(m, 1.4f);
    std::vector<float> actual(m * n, 0.0f);

    auto input = base_input();
    input.requested_score_mod = flashone::ScoreModKind::AdditiveBias;
    input.requested_bias_kind = flashone::BiasKind::BroadcastRow;
    const auto plan = flashone::make_runtime_plan(input,
                                                  /*one_dnn_available=*/true,
                                                  /*one_dnn_post_ops_available=*/true);
    require(plan.fallback_reason == flashone::FallbackReason::UnsupportedBroadcast,
            "broadcast bias should stay fallback in Stage 1.2");

    const flashone::StridedMatmulShape shape{/*m=*/m,
                                             /*n=*/n,
                                             /*k=*/d,
                                             /*a_stride_m=*/d,
                                             /*a_stride_k=*/1,
                                             /*b_stride_k=*/1,
                                             /*b_stride_n=*/d,
                                             /*c_stride_m=*/n,
                                             /*c_stride_n=*/1};
    flashone::QkScoreTilePostOpsInput post_ops;
    post_ops.additive_bias = row_bias.data();
    post_ops.additive_bias_stride_m = 1;
    post_ops.additive_bias_stride_n = 0;

    flashone::QkScoreTileDebugInfo debug;
    flashone::qk_score_tile_inplace(q.data(), k.data(), actual.data(), shape, plan, post_ops, &debug);
    const auto expected = reference_scores(q, k, row_bias.data(), 1, 0, 1.0f, false, m, n, d);
    require_close(expected, actual, 1e-6f, "broadcast fallback should preserve score semantics");
    require(debug.backend == flashone::QkBackendKind::Reference, "expected reference backend");
    require(debug.fallback_reason == flashone::FallbackReason::UnsupportedBroadcast,
            "expected UnsupportedBroadcast debug reason");
}

}  // namespace

int main() {
    test_scale_post_op();
    test_scale_additive_bias_post_op();
    test_reference_fallback_for_broadcast_bias();
    std::cout << "flashone QK score tile tests passed\n";
    return 0;
}
