#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "flashone/tile_kernel.hpp"

namespace flashone {

struct AttentionShape {
    std::size_t query_tokens;
    std::size_t key_tokens;
    std::size_t head_dim;
    std::size_t value_dim;
};

struct BlockMask {
    std::size_t query_block_size = 1;
    std::size_t key_block_size = 1;
    std::size_t query_blocks = 0;
    std::size_t key_blocks = 0;
    // Row-major [query_blocks, key_blocks]. Non-zero means the block is allowed.
    std::vector<std::uint8_t> allowed;

    bool allows(std::size_t query_index, std::size_t key_index) const;
};

using ScoreBiasFn = std::function<float(std::size_t query_index, std::size_t key_index)>;

struct AttentionOptions {
    float scale = 1.0f;
    bool causal = false;
    std::size_t key_block_size = 64;
    std::size_t query_block_size = 1;
    TileKernelKind qk_tile_kernel = TileKernelKind::Reference;
    const BlockMask* block_mask = nullptr;
    ScoreBiasFn score_bias = nullptr;
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

std::vector<float> flash_attention_q_tile(const std::vector<float>& q,
                                         const std::vector<float>& k,
                                         const std::vector<float>& v,
                                         const AttentionShape& shape,
                                         const AttentionOptions& options);

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b);

}  // namespace flashone
