#pragma once

#include "flashone/score_mod_plan.hpp"

#include <string>

namespace flashone {

enum class BlockMaskKind {
    None,
    Causal,
    Unsupported,
};

enum class QkBackendKind {
    Reference,
    OneDnnMatmul,
    OneDnnBrgemmBaseline,
};

enum class QkLayoutKind {
    RowMajorKStrided,
    CopiedTransposedK,
    BrgemmTransformedKBaseline,
};

struct BlockMaskPlan {
    BlockMaskKind kind{BlockMaskKind::None};
    bool has_boundary_tiles{false};
    bool can_skip_tiles{false};
    bool requires_mask_tile_generator{false};
    FallbackReason fallback_reason{FallbackReason::None};
    std::string signature{"none"};
};

struct MaskTileDescriptor {
    bool required{false};
    int q_start{0};
    int k_start{0};
    int q_size{0};
    int k_size{0};
    float masked_value{0.0f};
    bool all_masked_row_possible{false};
};

struct RuntimePlanInput {
    int batch{1};
    int heads{1};
    int query_length{0};
    int key_length{0};
    int head_dim{0};
    int value_dim{0};

    DataType q_dtype{DataType::F32};
    DataType k_dtype{DataType::F32};
    DataType v_dtype{DataType::F32};
    DataType out_dtype{DataType::F32};

    ScoreModKind requested_score_mod{ScoreModKind::None};
    bool has_scale{false};
    float scale_value{1.0f};
    BiasKind requested_bias_kind{BiasKind::None};

    BlockMaskKind requested_block_mask{BlockMaskKind::None};

    bool force_reference{false};
    bool enable_onednn{true};
    bool enable_debug{false};
};

struct RuntimePlanCacheKey {
    std::string semantic_shape;
    std::string dtype_signature;
    std::string score_mod_signature;
    std::string block_mask_signature;
    std::string policy_signature;
};

struct RuntimePlan {
    RuntimePlanInput input;

    ScoreModPlan score_mod_plan;
    BlockMaskPlan block_mask_plan;

    QkBackendKind qk_backend{QkBackendKind::Reference};
    QkLayoutKind qk_layout{QkLayoutKind::RowMajorKStrided};
    LoweringStatus qk_lowering_status{LoweringStatus::ReferenceFallback};

    bool uses_onednn_post_ops{false};
    bool requires_flashone_epilogue{false};
    bool requires_mask_tile_generator{false};

    FallbackReason fallback_reason{FallbackReason::None};

    std::string runtime_plan_signature;
    std::string debug_summary;
};

const char* to_string(BlockMaskKind value);
const char* to_string(QkBackendKind value);
const char* to_string(QkLayoutKind value);

BlockMaskPlan make_block_mask_plan(BlockMaskKind requested_kind,
                                   int query_length,
                                   int key_length);
MaskTileDescriptor make_causal_mask_tile_descriptor(int q_start,
                                                    int k_start,
                                                    int q_size,
                                                    int k_size);
RuntimePlan make_runtime_plan(const RuntimePlanInput& input,
                              bool one_dnn_available,
                              bool one_dnn_post_ops_available);
RuntimePlanCacheKey make_runtime_plan_cache_key(const RuntimePlan& plan);

}  // namespace flashone
