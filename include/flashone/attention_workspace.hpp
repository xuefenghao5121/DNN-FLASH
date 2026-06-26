#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace flashone {

/// Pre-allocated workspace buffers for attention kernels.
/// All buffers are resized once and reused across tile iterations,
/// eliminating per-iteration heap allocations.
struct AttentionWorkspace {
    // Tile buffers (reused across K-block iterations)
    std::vector<float> q_tile;       // [q_block, head_dim]
    std::vector<float> k_tile_t;     // [head_dim, k_block]  (transposed K)
    std::vector<float> v_tile;       // [k_block, value_dim]
    std::vector<float> score_tile;   // [q_block, k_block]
    std::vector<float> p_tile;       // [q_block, k_block]  (softmax weights)
    std::vector<float> pv_tile;      // [q_block, value_dim]

    // Per-query-row state (reused across Q-block iterations)
    std::vector<float> running_max;
    std::vector<float> running_sum;
    std::vector<float> acc;          // [q_block, value_dim]
    std::vector<float> old_scales;
    std::vector<float> new_maxes;
    std::vector<float> block_sums;
    std::vector<std::uint8_t> row_has_valid;

    /// Resize all buffers to fit the given tile dimensions.
    /// Safe to call repeatedly; only allocates if current capacity is insufficient.
    void resize(std::size_t q_block_size, std::size_t k_block_size,
                std::size_t head_dim, std::size_t value_dim);
};

}  // namespace flashone
