#include "onednn_flash/qk_score_tile_internal.hpp"

#ifdef ONEDNN_FLASH_HAS_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#endif

#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace onednn_flash {
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

#ifndef ONEDNN_FLASH_HAS_ONEDNN
void qk_score_tile_onednn_post_ops_inplace(const float*,
                                           const float*,
                                           float*,
                                           const StridedMatmulShape&,
                                           const RuntimePlan&,
                                           const QkScoreTilePostOpsInput&,
                                           const QkScoreTileExecuteOptions&) {
    throw std::runtime_error("OneDNNFlash was built without oneDNN support");
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

struct QkPostOpsPrimitiveKey {
    std::size_t m;
    std::size_t n;
    std::size_t k;
    std::size_t a_stride_m;
    std::size_t a_stride_k;
    std::size_t b_stride_k;
    std::size_t b_stride_n;
    std::size_t c_stride_m;
    std::size_t c_stride_n;
    bool has_scale;
    float scale_value;
    bool has_same_shape_bias;
    std::size_t bias_stride_m;
    std::size_t bias_stride_n;

    bool operator<(const QkPostOpsPrimitiveKey& other) const {
        return std::tie(m,
                        n,
                        k,
                        a_stride_m,
                        a_stride_k,
                        b_stride_k,
                        b_stride_n,
                        c_stride_m,
                        c_stride_n,
                        has_scale,
                        scale_value,
                        has_same_shape_bias,
                        bias_stride_m,
                        bias_stride_n) <
               std::tie(other.m,
                        other.n,
                        other.k,
                        other.a_stride_m,
                        other.a_stride_k,
                        other.b_stride_k,
                        other.b_stride_n,
                        other.c_stride_m,
                        other.c_stride_n,
                        other.has_scale,
                        other.scale_value,
                        other.has_same_shape_bias,
                        other.bias_stride_m,
                        other.bias_stride_n);
    }
};

struct CachedQkPostOpsPrimitive {
    dnnl::matmul primitive;
    dnnl::memory::desc q_md;
    dnnl::memory::desc k_md;
    dnnl::memory::desc score_md;
    dnnl::memory::desc bias_md;
    dnnl::memory q_mem;
    dnnl::memory k_mem;
    dnnl::memory score_mem;
    dnnl::memory bias_mem;
    std::unordered_map<int, dnnl::memory> execute_args;
    bool has_bias{false};
    int bias_arg{0};
};

struct ThreadLocalOneDnnQkContext {
    dnnl::engine engine{dnnl::engine::kind::cpu, 0};
    dnnl::stream stream{engine};
    std::map<QkPostOpsPrimitiveKey, CachedQkPostOpsPrimitive> primitive_cache;
    bool has_pending_work{false};

    // Cache observability counters
    std::size_t cache_hits{0};
    std::size_t cache_misses{0};
    std::size_t handle_rebinds{0};
    std::size_t immediate_waits{0};
    std::size_t deferred_waits{0};
};

ThreadLocalOneDnnQkContext& thread_local_qk_context() {
    thread_local ThreadLocalOneDnnQkContext context;
    return context;
}

QkPostOpsPrimitiveKey make_key(const StridedMatmulShape& shape,
                               const RuntimePlan& plan,
                               const QkScoreTilePostOpsInput& post_ops,
                               bool apply_bias) {
    return QkPostOpsPrimitiveKey{shape.m,
                                 shape.n,
                                 shape.k,
                                 shape.a_stride_m,
                                 shape.a_stride_k,
                                 shape.b_stride_k,
                                 shape.b_stride_n,
                                 shape.c_stride_m,
                                 shape.c_stride_n,
                                 plan.score_mod_plan.has_scale,
                                 plan.score_mod_plan.has_scale ? plan.score_mod_plan.scale_value : 1.0f,
                                 apply_bias,
                                 apply_bias ? post_ops.additive_bias_stride_m : 0,
                                 apply_bias ? post_ops.additive_bias_stride_n : 0};
}

CachedQkPostOpsPrimitive make_cached_primitive(const QkPostOpsPrimitiveKey& key,
                                               const RuntimePlan& plan,
                                               const dnnl::engine& engine) {
    CachedQkPostOpsPrimitive cached;
    cached.q_md = md2(key.m, key.k, key.a_stride_m, key.a_stride_k);
    cached.k_md = md2(key.k, key.n, key.b_stride_k, key.b_stride_n);
    cached.score_md = md2(key.m, key.n, key.c_stride_m, key.c_stride_n);

    dnnl::primitive_attr attr;
    dnnl::post_ops ops;

    if (key.has_scale) {
        ops.append_eltwise(dnnl::algorithm::eltwise_linear,
                           plan.score_mod_plan.scale_value,
                           0.0f);
    }
    if (key.has_same_shape_bias) {
        cached.bias_md = md2(key.m, key.n, key.bias_stride_m, key.bias_stride_n);
        ops.append_binary(dnnl::algorithm::binary_add, cached.bias_md);
        cached.has_bias = true;
        cached.bias_arg = DNNL_ARG_ATTR_MULTIPLE_POST_OP(key.has_scale ? 1 : 0) | DNNL_ARG_SRC_1;
    }
    attr.set_post_ops(ops);

    const auto pd = dnnl::matmul::primitive_desc(engine,
                                                cached.q_md,
                                                cached.k_md,
                                                cached.score_md,
                                                attr);
    cached.primitive = dnnl::matmul(pd);

    cached.q_mem = dnnl::memory(cached.q_md, engine, DNNL_MEMORY_NONE);
    cached.k_mem = dnnl::memory(cached.k_md, engine, DNNL_MEMORY_NONE);
    cached.score_mem = dnnl::memory(cached.score_md, engine, DNNL_MEMORY_NONE);
    if (cached.has_bias) {
        cached.bias_mem = dnnl::memory(cached.bias_md, engine, DNNL_MEMORY_NONE);
    }

    cached.execute_args.reserve(cached.has_bias ? 4 : 3);
    cached.execute_args.emplace(DNNL_ARG_SRC, cached.q_mem);
    cached.execute_args.emplace(DNNL_ARG_WEIGHTS, cached.k_mem);
    cached.execute_args.emplace(DNNL_ARG_DST, cached.score_mem);
    if (cached.has_bias) {
        cached.execute_args.emplace(cached.bias_arg, cached.bias_mem);
    }
    return cached;
}

CachedQkPostOpsPrimitive& get_cached_primitive(ThreadLocalOneDnnQkContext& context,
                                               const QkPostOpsPrimitiveKey& key,
                                               const RuntimePlan& plan) {
    auto it = context.primitive_cache.find(key);
    if (it == context.primitive_cache.end()) {
        it = context.primitive_cache.emplace(
            key, make_cached_primitive(key, plan, context.engine)).first;
        ++context.cache_misses;
    } else {
        ++context.cache_hits;
    }
    return it->second;
}

void bind_execute_memory_handles(CachedQkPostOpsPrimitive& cached,
                                 const float* q,
                                 const float* k,
                                 float* score,
                                 const QkScoreTilePostOpsInput& post_ops,
                                 ThreadLocalOneDnnQkContext& context) {
    cached.q_mem.set_data_handle(const_cast<float*>(q));
    cached.k_mem.set_data_handle(const_cast<float*>(k));
    cached.score_mem.set_data_handle(score);
    if (cached.has_bias) {
        cached.bias_mem.set_data_handle(const_cast<float*>(post_ops.additive_bias));
    }
    ++context.handle_rebinds;
}

void qk_score_tile_onednn_post_ops_inplace(const float* q,
                                           const float* k,
                                           float* score,
                                           const StridedMatmulShape& shape,
                                           const RuntimePlan& plan,
                                           const QkScoreTilePostOpsInput& post_ops,
                                           const QkScoreTileExecuteOptions& execute_options) {
    const bool apply_bias = has_same_shape_bias(plan, post_ops);
    if (plan.score_mod_plan.has_bias && !apply_bias) {
        throw std::invalid_argument("Stage 1.8 only supports same-shape additive bias tile post-op");
    }

    auto& context = thread_local_qk_context();
    const auto key = make_key(shape, plan, post_ops, apply_bias);
    auto& cached = get_cached_primitive(context, key, plan);
    bind_execute_memory_handles(cached, q, k, score, post_ops, context);

    cached.primitive.execute(context.stream, cached.execute_args);
    context.has_pending_work = true;
    if (execute_options.sync_mode == QkScoreTileSyncMode::WaitAfterExecute) {
        context.stream.wait();
        context.has_pending_work = false;
        ++context.immediate_waits;
    }
}
#endif

}  // namespace

void qk_score_tile_wait_for_onednn() {
#ifdef ONEDNN_FLASH_HAS_ONEDNN
    auto& context = thread_local_qk_context();
    if (context.has_pending_work) {
        context.stream.wait();
        context.has_pending_work = false;
        ++context.deferred_waits;
    }
#endif
}

QkScoreTileCacheStats qk_score_tile_get_cache_stats() {
    QkScoreTileCacheStats stats;
#ifdef ONEDNN_FLASH_HAS_ONEDNN
    auto& context = thread_local_qk_context();
    stats.primitive_cache_hits = context.cache_hits;
    stats.primitive_cache_misses = context.cache_misses;
    stats.memory_handle_rebinds = context.handle_rebinds;
    stats.immediate_waits = context.immediate_waits;
    stats.deferred_waits = context.deferred_waits;
    stats.cache_size = context.primitive_cache.size();
#endif
    return stats;
}

void qk_score_tile_reset_cache_stats() {
#ifdef ONEDNN_FLASH_HAS_ONEDNN
    auto& context = thread_local_qk_context();
    context.cache_hits = 0;
    context.cache_misses = 0;
    context.handle_rebinds = 0;
    context.immediate_waits = 0;
    context.deferred_waits = 0;
#endif
}

void qk_score_tile_inplace_with_options(const float* q,
                                        const float* k,
                                        float* score,
                                        const StridedMatmulShape& shape,
                                        const RuntimePlan& plan,
                                        const QkScoreTilePostOpsInput& post_ops,
                                        const QkScoreTileExecuteOptions& execute_options,
                                        QkScoreTileDebugInfo* debug) {
    validate_qk_score_tile_args(q, k, score, shape);

    const bool can_use_onednn_post_ops =
        plan.qk_backend == QkBackendKind::OneDnnMatmul &&
        plan.qk_lowering_status == LoweringStatus::LoweredToOneDnnPostOps &&
        plan.fallback_reason == FallbackReason::None;

    if (can_use_onednn_post_ops) {
        try {
            qk_score_tile_onednn_post_ops_inplace(q, k, score, shape, plan, post_ops, execute_options);
            set_debug(debug,
                      QkBackendKind::OneDnnMatmul,
                      LoweringStatus::LoweredToOneDnnPostOps,
                      FallbackReason::None,
                      plan.score_mod_plan.kind != ScoreModKind::None,
                      execute_options.sync_mode == QkScoreTileSyncMode::WaitAfterExecute
                          ? "qk_score_tile: dnnl::matmul + post_ops cached primitive"
                          : "qk_score_tile: dnnl::matmul + post_ops deferred stream wait");
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

void qk_score_tile_inplace(const float* q,
                           const float* k,
                           float* score,
                           const StridedMatmulShape& shape,
                           const RuntimePlan& plan,
                           const QkScoreTilePostOpsInput& post_ops,
                           QkScoreTileDebugInfo* debug) {
    qk_score_tile_inplace_with_options(q,
                                       k,
                                       score,
                                       shape,
                                       plan,
                                       post_ops,
                                       QkScoreTileExecuteOptions{},
                                       debug);
}

}  // namespace onednn_flash
