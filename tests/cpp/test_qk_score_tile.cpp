#include "onednn_flash/qk_score_tile_internal.hpp"

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

onednn_flash::RuntimePlanInput base_input() {
    onednn_flash::RuntimePlanInput input;
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

void run_qk_score_tile_case(onednn_flash::ScoreModKind kind,
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
    input.requested_bias_kind = has_bias ? onednn_flash::BiasKind::SameShapeTile : onednn_flash::BiasKind::None;
    const auto plan = onednn_flash::make_runtime_plan(input,
                                                  /*one_dnn_available=*/true,
                                                  /*one_dnn_post_ops_available=*/true);

    onednn_flash::QkScoreTilePostOpsInput post_ops;
    if (has_bias) {
        post_ops.additive_bias = bias.data();
        post_ops.additive_bias_stride_m = n;
        post_ops.additive_bias_stride_n = 1;
    }

    const onednn_flash::StridedMatmulShape shape{/*m=*/m,
                                             /*n=*/n,
                                             /*k=*/d,
                                             /*a_stride_m=*/d,
                                             /*a_stride_k=*/1,
                                             /*b_stride_k=*/1,
                                             /*b_stride_n=*/d,
                                             /*c_stride_m=*/n,
                                             /*c_stride_n=*/1};
    onednn_flash::QkScoreTileDebugInfo debug;
    onednn_flash::qk_score_tile_inplace(q.data(), k.data(), actual.data(), shape, plan, post_ops, &debug);

    const auto expected = reference_scores(q, k, has_bias ? bias.data() : nullptr, n, 1, scale, has_scale, m, n, d);
    require_close(expected, actual, 1e-5f, "QK score tile post-op mismatch");
#ifdef ONEDNN_FLASH_HAS_ONEDNN
    require(debug.backend == onednn_flash::QkBackendKind::OneDnnMatmul, "expected oneDNN matmul backend");
    require(debug.lowering_status == onednn_flash::LoweringStatus::LoweredToOneDnnPostOps,
            "expected oneDNN post-op lowering");
    require(debug.fallback_reason == onednn_flash::FallbackReason::None, "unexpected fallback reason");
#endif
}

void test_scale_post_op() {
    run_qk_score_tile_case(onednn_flash::ScoreModKind::Scale,
                           /*has_scale=*/true,
                           /*scale=*/0.25f,
                           /*has_bias=*/false);
}

void test_scale_additive_bias_post_op() {
    run_qk_score_tile_case(onednn_flash::ScoreModKind::ScaleAdditiveBias,
                           /*has_scale=*/true,
                           /*scale=*/0.5f,
                           /*has_bias=*/true);
}

void test_deferred_wait_mode_preserves_one_dnn_results() {
    constexpr std::size_t tile_count = 4;
    constexpr std::size_t m = 4;
    constexpr std::size_t n = 5;
    constexpr std::size_t d = 3;
    const auto q = make_values(tile_count * m * d, 0.11f);
    const auto k = make_values(tile_count * n * d, 0.71f);
    const auto bias = make_values(tile_count * m * n, 1.31f);
    std::vector<float> actual(tile_count * m * n, 0.0f);
    std::vector<float> expected(tile_count * m * n, 0.0f);

    auto input = base_input();
    input.requested_score_mod = onednn_flash::ScoreModKind::ScaleAdditiveBias;
    input.has_scale = true;
    input.scale_value = 0.5f;
    input.requested_bias_kind = onednn_flash::BiasKind::SameShapeTile;
    const auto plan = onednn_flash::make_runtime_plan(input,
                                                  /*one_dnn_available=*/true,
                                                  /*one_dnn_post_ops_available=*/true);

    const onednn_flash::StridedMatmulShape shape{/*m=*/m,
                                             /*n=*/n,
                                             /*k=*/d,
                                             /*a_stride_m=*/d,
                                             /*a_stride_k=*/1,
                                             /*b_stride_k=*/1,
                                             /*b_stride_n=*/d,
                                             /*c_stride_m=*/n,
                                             /*c_stride_n=*/1};
    onednn_flash::QkScoreTileExecuteOptions execute_options;
    execute_options.sync_mode = onednn_flash::QkScoreTileSyncMode::DeferUntilExplicitWait;

    onednn_flash::QkScoreTileDebugInfo debug;
    for (std::size_t tile = 0; tile < tile_count; ++tile) {
        const auto q_offset = tile * m * d;
        const auto k_offset = tile * n * d;
        const auto score_offset = tile * m * n;
        onednn_flash::QkScoreTilePostOpsInput post_ops;
        post_ops.additive_bias = bias.data() + score_offset;
        post_ops.additive_bias_stride_m = n;
        post_ops.additive_bias_stride_n = 1;
        onednn_flash::qk_score_tile_inplace_with_options(q.data() + q_offset,
                                                     k.data() + k_offset,
                                                     actual.data() + score_offset,
                                                     shape,
                                                     plan,
                                                     post_ops,
                                                     execute_options,
                                                     &debug);
        const auto expected_tile = reference_scores(std::vector<float>(q.begin() + static_cast<std::ptrdiff_t>(q_offset),
                                                                       q.begin() + static_cast<std::ptrdiff_t>(q_offset + m * d)),
                                                    std::vector<float>(k.begin() + static_cast<std::ptrdiff_t>(k_offset),
                                                                       k.begin() + static_cast<std::ptrdiff_t>(k_offset + n * d)),
                                                    bias.data() + score_offset,
                                                    n,
                                                    1,
                                                    0.5f,
                                                    true,
                                                    m,
                                                    n,
                                                    d);
        std::copy(expected_tile.begin(), expected_tile.end(), expected.begin() + static_cast<std::ptrdiff_t>(score_offset));
    }
    onednn_flash::qk_score_tile_wait_for_onednn();

    require_close(expected, actual, 1e-5f, "deferred wait QK score tile mismatch");
#ifdef ONEDNN_FLASH_HAS_ONEDNN
    require(debug.backend == onednn_flash::QkBackendKind::OneDnnMatmul, "expected deferred oneDNN backend");
    require(debug.lowering_status == onednn_flash::LoweringStatus::LoweredToOneDnnPostOps,
            "expected deferred oneDNN post-op lowering");
    require(debug.fallback_reason == onednn_flash::FallbackReason::None,
            "unexpected deferred oneDNN fallback reason");
#endif
}

void test_cache_observability_counters() {
    constexpr std::size_t m = 4;
    constexpr std::size_t n = 5;
    constexpr std::size_t d = 3;
    const auto q = make_values(m * d, 0.3f);
    const auto k = make_values(n * d, 0.9f);
    std::vector<float> actual(m * n, 0.0f);

    auto input = base_input();
    input.requested_score_mod = onednn_flash::ScoreModKind::Scale;
    input.has_scale = true;
    input.scale_value = 0.75f;  // unique scale to avoid cache hits from prior tests
    const auto plan = onednn_flash::make_runtime_plan(input,
                                                  /*one_dnn_available=*/true,
                                                  /*one_dnn_post_ops_available=*/true);

    const onednn_flash::StridedMatmulShape shape{/*m=*/m,
                                             /*n=*/n,
                                             /*k=*/d,
                                             /*a_stride_m=*/d,
                                             /*a_stride_k=*/1,
                                             /*b_stride_k=*/1,
                                             /*b_stride_n=*/d,
                                             /*c_stride_m=*/n,
                                             /*c_stride_n=*/1};
    onednn_flash::QkScoreTilePostOpsInput post_ops;
    onednn_flash::QkScoreTileDebugInfo debug;

    onednn_flash::qk_score_tile_reset_cache_stats();
    onednn_flash::qk_score_tile_inplace(q.data(), k.data(), actual.data(), shape, plan, post_ops, &debug);
    auto stats_after_first = onednn_flash::qk_score_tile_get_cache_stats();

#ifdef ONEDNN_FLASH_HAS_ONEDNN
    require(stats_after_first.primitive_cache_misses == 1, "first call should be cache miss");
    require(stats_after_first.primitive_cache_hits == 0, "first call should have no hits");
    require(stats_after_first.memory_handle_rebinds == 1, "first call should rebind once");
    require(stats_after_first.immediate_waits == 1, "first call should wait once");
    require(stats_after_first.deferred_waits == 0, "first call should not defer");
    require(stats_after_first.cache_size >= 1, "cache should have at least one entry");
#else
    require(stats_after_first.primitive_cache_misses == 0, "no oneDNN: no cache misses");
    require(stats_after_first.cache_size == 0, "no oneDNN: empty cache");
#endif

    onednn_flash::qk_score_tile_inplace(q.data(), k.data(), actual.data(), shape, plan, post_ops, &debug);
    auto stats_after_second = onednn_flash::qk_score_tile_get_cache_stats();

#ifdef ONEDNN_FLASH_HAS_ONEDNN
    require(stats_after_second.primitive_cache_misses == 1, "second call should still be 1 miss");
    require(stats_after_second.primitive_cache_hits == 1, "second call should be 1 hit");
    require(stats_after_second.memory_handle_rebinds == 2, "second call should rebind again");
    require(stats_after_second.immediate_waits == 2, "second call should wait again");
#endif

    onednn_flash::qk_score_tile_reset_cache_stats();
    auto stats_after_reset = onednn_flash::qk_score_tile_get_cache_stats();
    require(stats_after_reset.primitive_cache_hits == 0, "reset should zero hits");
    require(stats_after_reset.primitive_cache_misses == 0, "reset should zero misses");
    require(stats_after_reset.memory_handle_rebinds == 0, "reset should zero rebinds");
    require(stats_after_reset.immediate_waits == 0, "reset should zero immediate waits");
    require(stats_after_reset.deferred_waits == 0, "reset should zero deferred waits");
    // reset only zeros counters, does not clear the primitive cache itself
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
    input.requested_score_mod = onednn_flash::ScoreModKind::AdditiveBias;
    input.requested_bias_kind = onednn_flash::BiasKind::BroadcastRow;
    const auto plan = onednn_flash::make_runtime_plan(input,
                                                  /*one_dnn_available=*/true,
                                                  /*one_dnn_post_ops_available=*/true);
    require(plan.fallback_reason == onednn_flash::FallbackReason::UnsupportedBroadcast,
            "broadcast bias should stay fallback in Stage 1.2");

    const onednn_flash::StridedMatmulShape shape{/*m=*/m,
                                             /*n=*/n,
                                             /*k=*/d,
                                             /*a_stride_m=*/d,
                                             /*a_stride_k=*/1,
                                             /*b_stride_k=*/1,
                                             /*b_stride_n=*/d,
                                             /*c_stride_m=*/n,
                                             /*c_stride_n=*/1};
    onednn_flash::QkScoreTilePostOpsInput post_ops;
    post_ops.additive_bias = row_bias.data();
    post_ops.additive_bias_stride_m = 1;
    post_ops.additive_bias_stride_n = 0;

    onednn_flash::QkScoreTileDebugInfo debug;
    onednn_flash::qk_score_tile_inplace(q.data(), k.data(), actual.data(), shape, plan, post_ops, &debug);
    const auto expected = reference_scores(q, k, row_bias.data(), 1, 0, 1.0f, false, m, n, d);
    require_close(expected, actual, 1e-6f, "broadcast fallback should preserve score semantics");
    require(debug.backend == onednn_flash::QkBackendKind::Reference, "expected reference backend");
    require(debug.fallback_reason == onednn_flash::FallbackReason::UnsupportedBroadcast,
            "expected UnsupportedBroadcast debug reason");
}

}  // namespace

int main() {
    test_scale_post_op();
    test_scale_additive_bias_post_op();
    test_deferred_wait_mode_preserves_one_dnn_results();
    test_cache_observability_counters();
    test_reference_fallback_for_broadcast_bias();
    std::cout << "onednn_flash QK score tile tests passed\n";
    return 0;
}
