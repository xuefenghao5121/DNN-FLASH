#include "flashone/runtime_plan.hpp"

#include <limits>
#include <sstream>

namespace flashone {

const char* to_string(BlockMaskKind value) {
    switch (value) {
        case BlockMaskKind::None:
            return "none";
        case BlockMaskKind::Causal:
            return "causal";
        case BlockMaskKind::Unsupported:
            return "unsupported";
    }
    return "unknown";
}

const char* to_string(QkBackendKind value) {
    switch (value) {
        case QkBackendKind::Reference:
            return "reference";
        case QkBackendKind::OneDnnMatmul:
            return "onednn_matmul";
        case QkBackendKind::OneDnnBrgemmBaseline:
            return "onednn_brgemm_baseline";
    }
    return "unknown";
}

const char* to_string(QkLayoutKind value) {
    switch (value) {
        case QkLayoutKind::RowMajorKStrided:
            return "row_major_k_strided";
        case QkLayoutKind::CopiedTransposedK:
            return "copied_transposed_k";
        case QkLayoutKind::BrgemmTransformedKBaseline:
            return "brgemm_transformed_k_baseline";
    }
    return "unknown";
}

namespace {

bool all_dtypes_stage1_supported(const RuntimePlanInput& input) {
    return input.q_dtype == DataType::F32 && input.k_dtype == DataType::F32 &&
           input.v_dtype == DataType::F32 && input.out_dtype == DataType::F32;
}

bool shape_is_valid(const RuntimePlanInput& input) {
    return input.batch > 0 && input.heads > 0 && input.query_length > 0 && input.key_length > 0 &&
           input.head_dim > 0 && input.value_dim > 0;
}

std::string shape_signature(const RuntimePlanInput& input) {
    std::ostringstream oss;
    oss << "b" << input.batch << "h" << input.heads << "m" << input.query_length << "n"
        << input.key_length << "d" << input.head_dim << "dv" << input.value_dim;
    return oss.str();
}

std::string dtype_signature(const RuntimePlanInput& input) {
    std::ostringstream oss;
    oss << "q" << to_string(input.q_dtype) << "k" << to_string(input.k_dtype) << "v"
        << to_string(input.v_dtype) << "o" << to_string(input.out_dtype);
    return oss.str();
}

}  // namespace

BlockMaskPlan make_block_mask_plan(BlockMaskKind requested_kind,
                                   int query_length,
                                   int key_length) {
    BlockMaskPlan plan;
    plan.kind = requested_kind;

    if (requested_kind == BlockMaskKind::None) {
        plan.signature = "none";
        return plan;
    }

    if (requested_kind == BlockMaskKind::Causal) {
        plan.has_boundary_tiles = true;
        plan.can_skip_tiles = key_length > query_length;
        plan.requires_mask_tile_generator = true;
        plan.signature = "causal";
        return plan;
    }

    plan.kind = BlockMaskKind::Unsupported;
    plan.fallback_reason = FallbackReason::UnsupportedBlockMask;
    plan.signature = "unsupported:unsupported_block_mask";
    return plan;
}

MaskTileDescriptor make_causal_mask_tile_descriptor(int q_start,
                                                    int k_start,
                                                    int q_size,
                                                    int k_size) {
    MaskTileDescriptor descriptor;
    descriptor.required = true;
    descriptor.q_start = q_start;
    descriptor.k_start = k_start;
    descriptor.q_size = q_size;
    descriptor.k_size = k_size;
    descriptor.masked_value = -std::numeric_limits<float>::infinity();
    descriptor.all_masked_row_possible = k_start > q_start + q_size - 1;
    return descriptor;
}

RuntimePlan make_runtime_plan(const RuntimePlanInput& input,
                              bool one_dnn_available,
                              bool one_dnn_post_ops_available) {
    RuntimePlan plan;
    plan.input = input;

    const bool valid_shape = shape_is_valid(input);
    const bool supported_dtypes = all_dtypes_stage1_supported(input);
    const bool use_onednn = input.enable_onednn && one_dnn_available && !input.force_reference &&
                            valid_shape && supported_dtypes;

    plan.score_mod_plan = make_score_mod_plan(input.requested_score_mod,
                                              input.q_dtype,
                                              input.has_scale,
                                              input.scale_value,
                                              input.requested_bias_kind,
                                              use_onednn && one_dnn_post_ops_available);
    plan.block_mask_plan = make_block_mask_plan(input.requested_block_mask,
                                                input.query_length,
                                                input.key_length);

    if (!valid_shape) {
        plan.fallback_reason = FallbackReason::UnsupportedShape;
    } else if (!supported_dtypes) {
        plan.fallback_reason = FallbackReason::UnsupportedDType;
    } else if (input.force_reference) {
        plan.fallback_reason = FallbackReason::DebugForcedReference;
    } else if (!input.enable_onednn) {
        plan.fallback_reason = FallbackReason::OneDnnDisabled;
    } else if (!one_dnn_available) {
        plan.fallback_reason = FallbackReason::OneDnnUnavailable;
    } else if (plan.score_mod_plan.fallback_reason != FallbackReason::None) {
        plan.fallback_reason = plan.score_mod_plan.fallback_reason;
    } else if (plan.block_mask_plan.fallback_reason != FallbackReason::None) {
        plan.fallback_reason = plan.block_mask_plan.fallback_reason;
    } else {
        plan.fallback_reason = FallbackReason::None;
    }

    if (use_onednn && plan.score_mod_plan.lowering_status == LoweringStatus::LoweredToOneDnnPostOps &&
        plan.block_mask_plan.fallback_reason == FallbackReason::None) {
        plan.qk_backend = QkBackendKind::OneDnnMatmul;
        plan.qk_layout = QkLayoutKind::RowMajorKStrided;
        plan.qk_lowering_status = LoweringStatus::LoweredToOneDnnPostOps;
        plan.uses_onednn_post_ops = input.requested_score_mod != ScoreModKind::None;
    } else {
        plan.qk_backend = QkBackendKind::Reference;
        plan.qk_layout = QkLayoutKind::RowMajorKStrided;
        plan.qk_lowering_status = LoweringStatus::ReferenceFallback;
        plan.uses_onednn_post_ops = false;
    }

    plan.requires_mask_tile_generator = plan.block_mask_plan.requires_mask_tile_generator;
    plan.requires_flashone_epilogue = plan.requires_mask_tile_generator ||
                                      plan.score_mod_plan.lowering_status == LoweringStatus::FlashOneEpilogue;

    std::ostringstream sig;
    sig << shape_signature(input) << '|' << dtype_signature(input) << '|'
        << plan.score_mod_plan.signature << '|' << plan.block_mask_plan.signature << '|'
        << to_string(plan.qk_backend) << '|' << to_string(plan.qk_layout);
    plan.runtime_plan_signature = sig.str();

    std::ostringstream dbg;
    dbg << "RuntimePlan{"
        << "backend=" << to_string(plan.qk_backend) << ", layout=" << to_string(plan.qk_layout)
        << ", score_mod=" << plan.score_mod_plan.signature
        << ", block_mask=" << plan.block_mask_plan.signature
        << ", fallback=" << to_string(plan.fallback_reason) << '}';
    plan.debug_summary = dbg.str();

    return plan;
}

RuntimePlanCacheKey make_runtime_plan_cache_key(const RuntimePlan& plan) {
    RuntimePlanCacheKey key;
    key.semantic_shape = shape_signature(plan.input);
    key.dtype_signature = dtype_signature(plan.input);
    key.score_mod_signature = plan.score_mod_plan.signature;
    key.block_mask_signature = plan.block_mask_plan.signature;

    std::ostringstream policy;
    policy << "force_reference=" << (plan.input.force_reference ? "true" : "false")
           << ";enable_onednn=" << (plan.input.enable_onednn ? "true" : "false")
           << ";backend=" << to_string(plan.qk_backend) << ";layout=" << to_string(plan.qk_layout);
    key.policy_signature = policy.str();
    return key;
}

}  // namespace flashone
