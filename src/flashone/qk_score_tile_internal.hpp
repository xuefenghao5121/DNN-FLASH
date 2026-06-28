#pragma once

#include "flashone/runtime_plan.hpp"
#include "flashone/tile_kernel.hpp"

namespace flashone {

struct QkScoreTilePostOpsInput {
    const float* additive_bias{nullptr};
    std::size_t additive_bias_stride_m{0};
    std::size_t additive_bias_stride_n{1};
};

struct QkScoreTileDebugInfo {
    QkBackendKind backend{QkBackendKind::Reference};
    LoweringStatus lowering_status{LoweringStatus::ReferenceFallback};
    FallbackReason fallback_reason{FallbackReason::None};
    bool used_onednn_post_ops{false};
    std::string message;
};

enum class QkScoreTileSyncMode {
    WaitAfterExecute,
    DeferUntilExplicitWait,
};

struct QkScoreTileExecuteOptions {
    QkScoreTileSyncMode sync_mode{QkScoreTileSyncMode::WaitAfterExecute};
};

void qk_score_tile_wait_for_onednn();

void qk_score_tile_inplace_with_options(const float* q,
                                        const float* k,
                                        float* score,
                                        const StridedMatmulShape& shape,
                                        const RuntimePlan& plan,
                                        const QkScoreTilePostOpsInput& post_ops,
                                        const QkScoreTileExecuteOptions& execute_options,
                                        QkScoreTileDebugInfo* debug = nullptr);

void qk_score_tile_inplace(const float* q,
                           const float* k,
                           float* score,
                           const StridedMatmulShape& shape,
                           const RuntimePlan& plan,
                           const QkScoreTilePostOpsInput& post_ops,
                           QkScoreTileDebugInfo* debug = nullptr);

}  // namespace flashone
