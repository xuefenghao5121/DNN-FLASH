#include "flashone/batched_attention.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace flashone {

void flash_attention_batched_qk_pv_tile(const float* q,
                                        const float* k,
                                        const float* v,
                                        float* out,
                                        const BatchedAttentionShape& shape,
                                        const AttentionOptions& options) {
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
        throw std::invalid_argument("batched attention input/output pointers must be non-null");
    }
    if (shape.batch == 0 || shape.heads == 0 || shape.query_tokens == 0 ||
        shape.key_tokens == 0 || shape.head_dim == 0 || shape.value_dim == 0) {
        throw std::invalid_argument("batched attention dimensions must be non-zero");
    }

    const AttentionShape inner_shape{
        shape.query_tokens,
        shape.key_tokens,
        shape.head_dim,
        shape.value_dim,
    };

    const auto q_stride = shape.query_tokens * shape.head_dim;
    const auto k_stride = shape.key_tokens * shape.head_dim;
    const auto v_stride = shape.key_tokens * shape.value_dim;
    const auto o_stride = shape.query_tokens * shape.value_dim;
    const auto total = shape.batch * shape.heads;

    // Thread-local workspace: allocated once per thread, reused across all batch/head pairs.
    thread_local AttentionWorkspace ws;

    for (std::size_t bh = 0; bh < total; ++bh) {
        const float* q_base = q + bh * q_stride;
        const float* k_base = k + bh * k_stride;
        const float* v_base = v + bh * v_stride;
        float* o_base = out + bh * o_stride;

        // Zero-copy: pass raw pointers directly, workspace handles all tile buffers.
        flash_attention_qk_pv_tile_ws(q_base, k_base, v_base, o_base,
                                      inner_shape, options, ws);
    }
}

}  // namespace flashone
