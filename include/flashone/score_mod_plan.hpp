#pragma once

#include <string>

namespace flashone {

enum class DataType {
    F32,
    BF16,
    Unsupported,
};

enum class ScoreModKind {
    None,
    Scale,
    AdditiveBias,
    ScaleAdditiveBias,
    Unsupported,
};

enum class BiasKind {
    None,
    SameShapeTile,
    BroadcastRow,
    BroadcastCol,
    AlibiGenerated,
    Unsupported,
};

enum class LoweringStatus {
    LoweredToOneDnnPostOps,
    PartiallyLowered,
    FlashOneEpilogue,
    ReferenceFallback,
    Unsupported,
};

enum class FallbackReason {
    None,
    OneDnnDisabled,
    OneDnnUnavailable,
    UnsupportedDType,
    UnsupportedShape,
    UnsupportedScoreMod,
    UnsupportedBlockMask,
    UnsupportedPostOp,
    UnsupportedBiasLayout,
    UnsupportedBroadcast,
    UnsupportedEdgeTile,
    NumericalSafetyFallback,
    DebugForcedReference,
};

struct ScoreModPlan {
    ScoreModKind kind{ScoreModKind::None};
    DataType dtype{DataType::F32};

    bool has_scale{false};
    float scale_value{1.0f};

    BiasKind bias_kind{BiasKind::None};
    bool has_bias{false};

    LoweringStatus lowering_status{LoweringStatus::LoweredToOneDnnPostOps};
    FallbackReason fallback_reason{FallbackReason::None};

    std::string signature{"none"};
};

const char* to_string(DataType value);
const char* to_string(ScoreModKind value);
const char* to_string(BiasKind value);
const char* to_string(LoweringStatus value);
const char* to_string(FallbackReason value);

ScoreModPlan make_score_mod_plan(ScoreModKind requested_kind,
                                 DataType dtype,
                                 bool has_scale,
                                 float scale_value,
                                 BiasKind requested_bias_kind,
                                 bool one_dnn_post_ops_available);

}  // namespace flashone
