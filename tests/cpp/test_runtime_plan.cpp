#include "flashone/runtime_plan.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_eq(const std::string& actual, const std::string& expected, const char* message) {
    if (actual != expected) {
        throw std::runtime_error(std::string(message) + ": expected=" + expected + " actual=" + actual);
    }
}

flashone::RuntimePlanInput base_input() {
    flashone::RuntimePlanInput input;
    input.batch = 1;
    input.heads = 2;
    input.query_length = 128;
    input.key_length = 128;
    input.head_dim = 64;
    input.value_dim = 64;
    return input;
}

void test_scale_postops_plan() {
    auto input = base_input();
    input.requested_score_mod = flashone::ScoreModKind::Scale;
    input.has_scale = true;
    input.scale_value = 0.125f;

    const auto plan = flashone::make_runtime_plan(input, true, true);
    require(plan.qk_backend == flashone::QkBackendKind::OneDnnMatmul, "scale should use oneDNN matmul");
    require(plan.uses_onednn_post_ops, "scale should use oneDNN post-ops");
    require(plan.fallback_reason == flashone::FallbackReason::None, "scale should not fallback");
    require_eq(plan.score_mod_plan.signature, "scale:f32", "unexpected scale signature");
    require(plan.runtime_plan_signature.find("scale:f32") != std::string::npos,
            "runtime signature should include score mod");
}

void test_same_shape_bias_plan() {
    auto input = base_input();
    input.requested_score_mod = flashone::ScoreModKind::ScaleAdditiveBias;
    input.has_scale = true;
    input.scale_value = 0.125f;
    input.requested_bias_kind = flashone::BiasKind::SameShapeTile;

    const auto plan = flashone::make_runtime_plan(input, true, true);
    require(plan.qk_backend == flashone::QkBackendKind::OneDnnMatmul,
            "same-shape bias should use oneDNN matmul when available");
    require(plan.uses_onednn_post_ops, "same-shape bias should use oneDNN post-ops");
    require(plan.fallback_reason == flashone::FallbackReason::None,
            "same-shape bias should not fallback when post-ops available");
    require_eq(plan.score_mod_plan.signature,
               "scale_additive_bias:same_shape_tile:f32",
               "unexpected bias signature");
}

void test_broadcast_bias_fallback() {
    auto input = base_input();
    input.requested_score_mod = flashone::ScoreModKind::AdditiveBias;
    input.requested_bias_kind = flashone::BiasKind::BroadcastRow;

    const auto plan = flashone::make_runtime_plan(input, true, true);
    require(plan.qk_backend == flashone::QkBackendKind::Reference,
            "broadcast bias is not Stage 1 fast path");
    require(plan.fallback_reason == flashone::FallbackReason::UnsupportedBroadcast,
            "broadcast bias fallback reason mismatch");
    require(plan.debug_summary.find("unsupported_broadcast") != std::string::npos,
            "debug summary should include fallback reason");
}

void test_causal_mask_requires_epilogue() {
    auto input = base_input();
    input.requested_block_mask = flashone::BlockMaskKind::Causal;

    const auto plan = flashone::make_runtime_plan(input, true, true);
    require(plan.qk_backend == flashone::QkBackendKind::OneDnnMatmul,
            "causal plan can still use oneDNN QK matmul");
    require(plan.requires_mask_tile_generator, "causal plan should require mask tile generator");
    require(plan.requires_flashone_epilogue, "causal boundary should stay in FlashOne epilogue");
    require_eq(plan.block_mask_plan.signature, "causal", "unexpected causal signature");

    const auto descriptor = flashone::make_causal_mask_tile_descriptor(0, 64, 32, 32);
    require(descriptor.required, "causal descriptor should be required");
    require(descriptor.all_masked_row_possible, "tile entirely above diagonal can contain all-masked rows");
}

void test_cache_key_is_semantic() {
    auto input = base_input();
    input.requested_score_mod = flashone::ScoreModKind::Scale;
    input.has_scale = true;
    const auto plan = flashone::make_runtime_plan(input, true, true);
    const auto key = flashone::make_runtime_plan_cache_key(plan);

    require_eq(key.semantic_shape, "b1h2m128n128d64dv64", "semantic shape mismatch");
    require_eq(key.dtype_signature, "qf32kf32vf32of32", "dtype signature mismatch");
    require_eq(key.score_mod_signature, "scale:f32", "score mod key mismatch");
    require_eq(key.block_mask_signature, "none", "block mask key mismatch");
    require(key.policy_signature.find("backend=onednn_matmul") != std::string::npos,
            "policy key should include selected backend");
}

void test_forced_reference_reason() {
    auto input = base_input();
    input.force_reference = true;
    const auto plan = flashone::make_runtime_plan(input, true, true);
    require(plan.qk_backend == flashone::QkBackendKind::Reference, "forced reference should select reference");
    require(plan.fallback_reason == flashone::FallbackReason::DebugForcedReference,
            "forced reference reason mismatch");
}

}  // namespace

int main() {
    test_scale_postops_plan();
    test_same_shape_bias_plan();
    test_broadcast_bias_fallback();
    test_causal_mask_requires_epilogue();
    test_cache_key_is_semantic();
    test_forced_reference_reason();
    std::cout << "flashone runtime plan tests passed\n";
    return 0;
}
