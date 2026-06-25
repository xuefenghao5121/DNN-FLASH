#pragma once

#include <cstddef>
#include <vector>

namespace flashone {

struct AttentionShape {
    std::size_t query_tokens;
    std::size_t key_tokens;
    std::size_t head_dim;
    std::size_t value_dim;
};

struct AttentionOptions {
    float scale = 1.0f;
    bool causal = false;
    std::size_t key_block_size = 64;
};

std::vector<float> standard_attention(const std::vector<float>& q,
                                      const std::vector<float>& k,
                                      const std::vector<float>& v,
                                      const AttentionShape& shape,
                                      const AttentionOptions& options);

std::vector<float> flash_attention_tiled(const std::vector<float>& q,
                                         const std::vector<float>& k,
                                         const std::vector<float>& v,
                                         const AttentionShape& shape,
                                         const AttentionOptions& options);

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b);

}  // namespace flashone
