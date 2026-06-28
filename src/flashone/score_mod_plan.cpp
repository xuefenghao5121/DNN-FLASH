#include "flashone/score_mod_plan.hpp"

#include <sstream>

namespace flashone {

const char* to_string(DataType value) {
    switch (value) {
        case DataType::F32:
            return "f32";
        case DataType::BF16:
            return "bf16";
        case DataType::Unsupported:
            return "unsupported";
    }
    return "unknown";
}

const char* to_string(ScoreModKind value) {
    switch (value) {
        case ScoreModKind::None:
            return "none";
        case ScoreModKind::Scale:
            return "scale";
        case ScoreModKind::AdditiveBias:
            return "additive_bias";
        case ScoreModKind::ScaleAdditiveBias:
            return "scale_additive_bias";
        case ScoreModKind::Unsupported:
            return "unsupported";
    }
    return "unknown";
}

const char* to_string(BiasKind value) {
    switch (value) {
        case BiasKind::None:
            return "none";
        case BiasKind::SameShapeTile:
            return "same_shape_tile";
        case BiasKind::BroadcastRow:
            return "broadcast_row";
        case BiasKind::BroadcastCol:
            return "broadcast_col";
        case BiasKind::AlibiGenerated:
            return "alibi_generated";
        case BiasKind::Unsupported:
            return "unsupported";
    }
    return "unknown";
}

const char* to_string(LoweringStatus value) {
    switch (value) {
        case LoweringStatus::LoweredToOneDnnPostOps:
            return "lowered_to_onednn_post_ops";
        case LoweringStatus::PartiallyLowered:
            return "partially_lowered";
        case LoweringStatus::FlashOneEpilogue:
            return "flashone_epilogue";
        case LoweringStatus::ReferenceFallback:
            return "reference_fallback";
        case LoweringStatus::Unsupported:
            return "unsupported";
    }
    return "unknown";
}

const char* to_string(FallbackReason value) {
    switch (value) {
        case FallbackReason::None:
            return "none";
        case FallbackReason::OneDnnDisabled:
            return "onednn_disabled";
        case FallbackReason::OneDnnUnavailable:
            return "onednn_unavailable";
        case FallbackReason::UnsupportedDType:
            return "unsupported_dtype";
        case FallbackReason::UnsupportedShape:
            return "unsupported_shape";
        case FallbackReason::UnsupportedScoreMod:
            return "unsupported_score_mod";
        case FallbackReason::UnsupportedBlockMask:
            return "unsupported_block_mask";
        case FallbackReason::UnsupportedPostOp:
            return "unsupported_post_op";
        case FallbackReason::UnsupportedBiasLayout:
            return "unsupported_bias_layout";
        case FallbackReason::UnsupportedBroadcast:
            return "unsupported_broadcast";
        case FallbackReason::UnsupportedEdgeTile:
            return "unsupported_edge_tile";
        case FallbackReason::NumericalSafetyFallback:
            return "numerical_safety_fallback";
        case FallbackReason::DebugForcedReference:
            return "debug_forced_reference";
    }
    return "unknown";
}

namespace {

bool dtype_is_stage1_supported(DataType dtype) {
    return dtype == DataType::F32;
}

std::string unsupported_signature(FallbackReason reason) {
    std::ostringstream oss;
    oss << "unsupported:" << to_string(reason);
    return oss.str();
}

}  // namespace

ScoreModPlan make_score_mod_plan(ScoreModKind requested_kind,
                                 DataType dtype,
                                 bool has_scale,
                                 float scale_value,
                                 BiasKind requested_bias_kind,
                                 bool one_dnn_post_ops_available) {
    ScoreModPlan plan;
    plan.kind = requested_kind;
    plan.dtype = dtype;
    plan.has_scale = has_scale;
    plan.scale_value = has_scale ? scale_value : 1.0f;
    plan.bias_kind = requested_bias_kind;
    plan.has_bias = requested_bias_kind != BiasKind::None;

    if (!dtype_is_stage1_supported(dtype)) {
        plan.kind = ScoreModKind::Unsupported;
        plan.lowering_status = LoweringStatus::ReferenceFallback;
        plan.fallback_reason = FallbackReason::UnsupportedDType;
        plan.signature = unsupported_signature(plan.fallback_reason);
        return plan;
    }

    if (requested_kind == ScoreModKind::Unsupported) {
        plan.lowering_status = LoweringStatus::ReferenceFallback;
        plan.fallback_reason = FallbackReason::UnsupportedScoreMod;
        plan.signature = unsupported_signature(plan.fallback_reason);
        return plan;
    }

    if (plan.has_bias && requested_bias_kind != BiasKind::SameShapeTile) {
        plan.lowering_status = LoweringStatus::FlashOneEpilogue;
        plan.fallback_reason = requested_bias_kind == BiasKind::BroadcastRow ||
                                       requested_bias_kind == BiasKind::BroadcastCol
                                   ? FallbackReason::UnsupportedBroadcast
                                   : FallbackReason::UnsupportedBiasLayout;
        plan.signature = unsupported_signature(plan.fallback_reason);
        return plan;
    }

    if (!one_dnn_post_ops_available && requested_kind != ScoreModKind::None) {
        plan.lowering_status = LoweringStatus::FlashOneEpilogue;
        plan.fallback_reason = FallbackReason::UnsupportedPostOp;
    } else {
        plan.lowering_status = LoweringStatus::LoweredToOneDnnPostOps;
        plan.fallback_reason = FallbackReason::None;
    }

    std::ostringstream oss;
    switch (requested_kind) {
        case ScoreModKind::None:
            oss << "none";
            break;
        case ScoreModKind::Scale:
            oss << "scale:" << to_string(dtype);
            break;
        case ScoreModKind::AdditiveBias:
            oss << "additive_bias:" << to_string(requested_bias_kind) << ':' << to_string(dtype);
            break;
        case ScoreModKind::ScaleAdditiveBias:
            oss << "scale_additive_bias:" << to_string(requested_bias_kind) << ':' << to_string(dtype);
            break;
        case ScoreModKind::Unsupported:
            oss << unsupported_signature(plan.fallback_reason);
            break;
    }
    plan.signature = oss.str();
    return plan;
}

}  // namespace flashone
