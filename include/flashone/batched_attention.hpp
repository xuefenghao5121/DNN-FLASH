#pragma once

#include <cstddef>

#include "flashone/attention.hpp"

namespace flashone {

struct BatchedAttentionShape {
    std::size_t batch;
    std::size_t heads;
    std::size_t query_tokens;
    std::size_t key_tokens;
    std::size_t head_dim;
    std::size_t value_dim;
};

// Flat row-major tensors:
// Q: [B,H,M,D]
// K: [B,H,N,D]
// V: [B,H,N,Dv]
// O: [B,H,M,Dv]
//
// Uses a thread-local AttentionWorkspace internally: zero per-call allocation
// after the first invocation. Operates directly on raw pointers (no vector copies).
void flash_attention_batched_qk_pv_tile(const float* q,
                                        const float* k,
                                        const float* v,
                                        float* out,
                                        const BatchedAttentionShape& shape,
                                        const AttentionOptions& options);

}  // namespace flashone
