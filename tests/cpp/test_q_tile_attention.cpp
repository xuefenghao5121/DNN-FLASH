#include "onednn_flash/attention.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> make_values(std::size_t size, float offset) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.07f + offset) * 0.35f;
    }
    return values;
}

void require_close(const std::vector<float>& expected,
                   const std::vector<float>& actual,
                   float tolerance,
                   const char* label) {
    const float diff = onednn_flash::max_abs_diff(expected, actual);
    if (diff > tolerance) {
        std::cerr << label << " diff=" << diff << " > " << tolerance << "\n";
        throw std::runtime_error(label);
    }
}

onednn_flash::BlockMask make_band_mask(std::size_t query_tokens,
                                   std::size_t key_tokens,
                                   std::size_t query_block,
                                   std::size_t key_block) {
    const auto qb = (query_tokens + query_block - 1) / query_block;
    const auto kb = (key_tokens + key_block - 1) / key_block;
    onednn_flash::BlockMask mask;
    mask.query_block_size = query_block;
    mask.key_block_size = key_block;
    mask.query_blocks = qb;
    mask.key_blocks = kb;
    mask.allowed.resize(qb * kb, 0);
    for (std::size_t i = 0; i < qb; ++i) {
        for (std::size_t j = 0; j < kb; ++j) {
            if (j <= i + 1) {
                mask.allowed[i * kb + j] = 1;
            }
        }
    }
    return mask;
}

void test_q_tile_matches_standard(onednn_flash::TileKernelKind kernel_kind) {
    const onednn_flash::AttentionShape shape{10, 13, 6, 5};
    const auto q = make_values(shape.query_tokens * shape.head_dim, 0.11f);
    const auto k = make_values(shape.key_tokens * shape.head_dim, 1.11f);
    const auto v = make_values(shape.key_tokens * shape.value_dim, 2.11f);
    auto mask = make_band_mask(shape.query_tokens, shape.key_tokens, 3, 4);

    onednn_flash::AttentionOptions options;
    options.scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    options.causal = true;
    options.key_block_size = 4;
    options.query_block_size = 3;
    options.qk_tile_kernel = kernel_kind;
    options.block_mask = &mask;
    options.score_bias = [](std::size_t qi, std::size_t kj) {
        return static_cast<float>(qi) * 0.005f - static_cast<float>(kj) * 0.003f;
    };

    const auto expected = onednn_flash::standard_attention(q, k, v, shape, options);
    const auto actual = onednn_flash::flash_attention_q_tile(q, k, v, shape, options);
    require_close(expected, actual, 1e-5f, onednn_flash::tile_kernel_name(kernel_kind));
}

}  // namespace

int main() {
    test_q_tile_matches_standard(onednn_flash::TileKernelKind::Reference);
#ifdef ONEDNN_FLASH_HAS_ONEDNN
    test_q_tile_matches_standard(onednn_flash::TileKernelKind::OneDnn);
#endif
    std::cout << "onednn_flash Q-tile attention tests passed\n";
    return 0;
}
