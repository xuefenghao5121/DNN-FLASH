#include "flashone/qk_score_tile_internal.hpp"

#ifdef FLASHONE_HAS_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#endif

#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace flashone {
namespace {

void validate_qk_score_tile_args(const float* q,
                                 const float* k,
                                 float* score,
                                 const StridedMatmulShape& shape) {
    if (shape.m == 0 || shape.n == 0 || shape.k == 0) {
        throw std::invalid_argument("QK score tile dimensions must be non-zero");
    }
    if (q == nullptr || k == nullptr || score == nullptr) {
        throw std::invalid_argument("QK score tile pointers must be non-null");
    }
}

bool has_same_shape_bias(const RuntimePlan& plan,
                         const QkScoreTilePostOpsInput& post_ops) {
    return plan.score_mod_plan.has_bias &&
           plan.score_mod_plan.bias_kind == BiasKind::SameShapeTile &&
           post_ops.additive_bias != nullptr;
}

float bias_value(const RuntimePlan& plan,
                 const QkScoreTilePostOpsInput& post_ops,
                 std::size_t i,
                 std::size_t j) {
    if (!plan.score_mod_plan.has_bias) {
        return 0.0f;
    }
    if (post_ops.additive_bias == nullptr) {
        throw std::invalid_argument("QK score tile additive bias pointer is required by plan");
    }
    return post_ops.additive_bias[i * post_ops.additive_bias_stride_m +
                                  j * post_ops.additive_bias_stride_n];
}

void apply_reference_post_ops(float* score,
                              const StridedMatmulShape& shape,
                              const RuntimePlan& plan,
                              const QkScoreTilePostOpsInput& post_ops) {
    const bool apply_scale = plan.score_mod_plan.has_scale;
    const bool apply_bias = plan.score_mod_plan.has_bias;
    for (std::size_t i = 0; i < shape.m; ++i) {
        for (std::size_t j = 0; j < shape.n; ++j) {
            const auto score_index = i * shape.c_stride_m + j * shape.c_stride_n;
            float value = score[score_index];
            if (apply_scale) {
                value *= plan.score_mod_plan.scale_value;
            }
            if (apply_bias) {
                value += bias_value(plan, post_ops, i, j);
            }
            score[score_index] = value;
        }
    }
}

void set_debug(QkScoreTileDebugInfo* debug,
               QkBackendKind backend,
               LoweringStatus status,
               FallbackReason reason,
               bool used_post_ops,
               const std::string& message) {
    if (debug == nullptr) {
        return;
    }
    debug->backend = backend;
    debug->lowering_status = status;
    debug->fallback_reason = reason;
    debug->used_onednn_post_ops = used_post_ops;
    debug->message = message;
}

#ifndef FLASHONE_HAS_ONEDNN
void qk_score_tile_onednn_post_ops_inplace(const float*,
                                           const float*,
                                           float*,
                                           const StridedMatmulShape&,
                                           const RuntimePlan&,
                                           const QkScoreTilePostOpsInput&) {
    throw std::runtime_error("FlashOne was built without oneDNN support");
}
#else

dnnl::memory::desc md2(std::size_t dim0,
                       std::size_t dim1,
                       std::size_t stride0,
                       std::size_t stride1) {
    return dnnl::memory::desc({static_cast<dnnl::memory::dim>(dim0),
                               static_cast<dnnl::memory::dim>(dim1)},
                              dnnl::memory::data_type::f32,
                              {static_cast<dnnl::memory::dim>(stride0),
                               static_cast<dnnl::memory::dim>(stride1)});
}

void qk_score_tile_onednn_post_ops_inplace(const float* q,
                                           const float* k,
                                           float* score,
                                           const StridedMatmulShape& shape,
                                           const RuntimePlan& plan,
                                           const QkScoreTilePostOpsInput& post_ops) {
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream stream(engine);

    const auto q_md = md2(shape.m, shape.k, shape.a_stride_m, shape.a_stride_k);
    const auto k_md = md2(shape.k, shape.n, shape.b_stride_k, shape.b_stride_n);
    const auto score_md = md2(shape.m, shape.n, shape.c_stride_m, shape.c_stride_n);

    dnnl::primitive_attr attr;
    dnnl::post_ops ops;

    const bool apply_scale = plan.score_mod_plan.has_scale;
    const bool apply_bias = has_same_shape_bias(plan, post_ops);

    if (apply_scale) {
        ops.append_eltwise(dnnl::algorithm::eltwise_linear,
                           plan.score_mod_plan.scale_value,
                           0.0f);
    }
    if (plan.score_mod_plan.has_bias && !apply_bias) {
        throw std::invalid_argument("Stage 1.2 only supports same-shape additive bias tile post-op");
    }
    if (apply_bias) {
        const auto bias_md = md2(shape.m,
                                 shape.n,
                                 post_ops.additive_bias_stride_m,
                                 post_ops.additive_bias_stride_n);
        ops.append_binary(dnnl::algorithm::binary_add, bias_md);
    }
    attr.set_post_ops(ops);

    const auto pd = dnnl::matmul::primitive_desc(engine, q_md, k_md, score_md, attr);
    const dnnl::matmul primitive(pd);

    dnnl::memory q_mem(q_md, engine, const_cast<float*>(q));
    dnnl::memory k_mem(k_md, engine, const_cast<float*>(k));
    dnnl::memory score_mem(score_md, engine, score);

    std::unordered_map<int, dnnl::memory> args{{DNNL_ARG_SRC, q_mem},
                                               {DNNL_ARG_WEIGHTS, k_mem},
                                               {DNNL_ARG_DST, score_mem}};
    dnnl::memory bias_mem;
    if (apply_bias) {
        const auto bias_md = md2(shape.m,
                                 shape.n,
                                 post_ops.additive_bias_stride_m,
                                 post_ops.additive_bias_stride_n);
        bias_mem = dnnl::memory(bias_md, engine, const_cast<float*>(post_ops.additive_bias));
        const int bias_arg = DNNL_ARG_ATTR_MULTIPLE_POST_OP(apply_scale ? 1 : 0) | DNNL_ARG_SRC_1;
        args.emplace(bias_arg, bias_mem);
    }

    primitive.execute(stream, args);
    stream.wait();
}
#endif

}  // namespace

void qk_score_tile_inplace(const float* q,
                           const float* k,
                           float* score,
                           const StridedMatmulShape& shape,
                           const RuntimePlan& plan,
                           const QkScoreTilePostOpsInput& post_ops,
                           QkScoreTileDebugInfo* debug) {
    validate_qk_score_tile_args(q, k, score, shape);

    const bool can_use_onednn_post_ops =
        plan.qk_backend == QkBackendKind::OneDnnMatmul &&
        plan.qk_lowering_status == LoweringStatus::LoweredToOneDnnPostOps &&
        plan.fallback_reason == FallbackReason::None;

    if (can_use_onednn_post_ops) {
        try {
            qk_score_tile_onednn_post_ops_inplace(q, k, score, shape, plan, post_ops);
            set_debug(debug,
                      QkBackendKind::OneDnnMatmul,
                      LoweringStatus::LoweredToOneDnnPostOps,
                      FallbackReason::None,
                      plan.score_mod_plan.kind != ScoreModKind::None,
                      "qk_score_tile: dnnl::matmul + post_ops");
            return;
        } catch (const std::exception& e) {
            matmul_tile_strided_inplace(TileKernelKind::Reference, q, k, score, shape);
            apply_reference_post_ops(score, shape, plan, post_ops);
            set_debug(debug,
                      QkBackendKind::Reference,
                      LoweringStatus::ReferenceFallback,
                      FallbackReason::UnsupportedPostOp,
                      false,
                      std::string("qk_score_tile: oneDNN post-op fallback: ") + e.what());
            return;
        }
    }

    matmul_tile_strided_inplace(TileKernelKind::Reference, q, k, score, shape);
    apply_reference_post_ops(score, shape, plan, post_ops);
    set_debug(debug,
              QkBackendKind::Reference,
              LoweringStatus::ReferenceFallback,
              plan.fallback_reason,
              false,
              "qk_score_tile: reference fallback");
}

}  // namespace flashone
