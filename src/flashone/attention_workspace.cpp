#include "flashone/attention_workspace.hpp"

namespace flashone {

void AttentionWorkspace::resize(std::size_t q_block_size, std::size_t k_block_size,
                                 std::size_t head_dim, std::size_t value_dim) {
    q_tile.resize(q_block_size * head_dim);
    k_tile_t.resize(head_dim * k_block_size);
    v_tile.resize(k_block_size * value_dim);
    score_tile.resize(q_block_size * k_block_size);
    p_tile.resize(q_block_size * k_block_size);
    pv_tile.resize(q_block_size * value_dim);
    running_max.resize(q_block_size);
    running_sum.resize(q_block_size);
    acc.resize(q_block_size * value_dim);
    old_scales.resize(q_block_size);
    new_maxes.resize(q_block_size);
    block_sums.resize(q_block_size);
    row_has_valid.resize(q_block_size);
}

}  // namespace flashone
