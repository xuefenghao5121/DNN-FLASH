#include "onednn_flash/attention.hpp"
#include "onednn_flash/attention_workspace.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> make_values(std::size_t size, float offset) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::cos(static_cast<float>(i) * 0.09f + offset) * 0.31f;
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

onednn_flash::BlockMask make_window_mask(std::size_t query_tokens,
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
            if (j + 1 >= i && j <= i + 1) {
                mask.allowed[i * kb + j] = 1;
            }
        }
    }
    return mask;
}

void test_ws_matches_standard(onednn_flash::TileKernelKind qk_kind,
                              onednn_flash::TileKernelKind pv_kind,
                              onednn_flash::QkTileLayout qk_layout = onednn_flash::QkTileLayout::StridedK) {
    const onednn_flash::AttentionShape shape{11, 14, 7, 6};
    const auto q = make_values(shape.query_tokens * shape.head_dim, 0.17f);
    const auto k = make_values(shape.key_tokens * shape.head_dim, 1.17f);
    const auto v = make_values(shape.key_tokens * shape.value_dim, 2.17f);
    auto mask = make_window_mask(shape.query_tokens, shape.key_tokens, 3, 4);

    onednn_flash::AttentionOptions options;
    options.scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    options.causal = false;
    options.key_block_size = 4;
    options.query_block_size = 3;
    options.qk_tile_kernel = qk_kind;
    options.pv_tile_kernel = pv_kind;
    options.qk_tile_layout = qk_layout;
    options.block_mask = &mask;
    options.score_bias = [](std::size_t qi, std::size_t kj) {
        return static_cast<float>(qi) * 0.004f - static_cast<float>(kj) * 0.002f;
    };

    const auto expected = onednn_flash::standard_attention(q, k, v, shape, options);

    std::vector<float> output(shape.query_tokens * shape.value_dim, 0.0f);
    onednn_flash::AttentionWorkspace ws;
    onednn_flash::flash_attention_qk_pv_tile_ws(q.data(), k.data(), v.data(), output.data(),
                                            shape, options, ws);
    require_close(expected, output, 1e-5f, "workspace attention mismatch");
}

void test_ws_matches_vector_api(onednn_flash::TileKernelKind qk_kind,
                                onednn_flash::TileKernelKind pv_kind,
                                onednn_flash::QkTileLayout qk_layout = onednn_flash::QkTileLayout::StridedK) {
    const onednn_flash::AttentionShape shape{32, 32, 64, 64};
    const auto q = make_values(shape.query_tokens * shape.head_dim, 0.3f);
    const auto k = make_values(shape.key_tokens * shape.head_dim, 1.5f);
    const auto v = make_values(shape.key_tokens * shape.value_dim, 2.8f);

    onednn_flash::AttentionOptions options;
    options.scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    options.causal = true;
    options.key_block_size = 32;
    options.query_block_size = 16;
    options.qk_tile_kernel = qk_kind;
    options.pv_tile_kernel = pv_kind;
    options.qk_tile_layout = qk_layout;

    const auto expected = onednn_flash::flash_attention_qk_pv_tile(q, k, v, shape, options);

    std::vector<float> output(shape.query_tokens * shape.value_dim, 0.0f);
    onednn_flash::AttentionWorkspace ws;
    onednn_flash::flash_attention_qk_pv_tile_ws(q.data(), k.data(), v.data(), output.data(),
                                            shape, options, ws);
    require_close(expected, output, 1e-5f, "workspace vs vector API mismatch");
}

void test_ws_reuse() {
    // Verify that calling ws-based attention twice gives correct results
    // (workspace buffers are properly reset each call).
    const onednn_flash::AttentionShape shape{16, 16, 32, 32};
    const auto q = make_values(shape.query_tokens * shape.head_dim, 0.1f);
    const auto k = make_values(shape.key_tokens * shape.head_dim, 0.2f);
    const auto v = make_values(shape.key_tokens * shape.value_dim, 0.3f);

    onednn_flash::AttentionOptions options;
    options.scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    options.causal = true;
    options.key_block_size = 16;
    options.query_block_size = 8;
    options.qk_tile_kernel = onednn_flash::TileKernelKind::Reference;
    options.pv_tile_kernel = onednn_flash::TileKernelKind::Reference;

    std::vector<float> out1(shape.query_tokens * shape.value_dim, 0.0f);
    std::vector<float> out2(shape.query_tokens * shape.value_dim, 0.0f);
    onednn_flash::AttentionWorkspace ws;

    onednn_flash::flash_attention_qk_pv_tile_ws(q.data(), k.data(), v.data(), out1.data(),
                                            shape, options, ws);
    onednn_flash::flash_attention_qk_pv_tile_ws(q.data(), k.data(), v.data(), out2.data(),
                                            shape, options, ws);

    require_close(out1, out2, 0.0f, "workspace reuse mismatch");
}

}  // namespace

int main() {
    test_ws_matches_standard(onednn_flash::TileKernelKind::Reference,
                             onednn_flash::TileKernelKind::Reference);
    test_ws_matches_vector_api(onednn_flash::TileKernelKind::Reference,
                               onednn_flash::TileKernelKind::Reference);
    test_ws_reuse();

#ifdef ONEDNN_FLASH_HAS_ONEDNN
    test_ws_matches_standard(onednn_flash::TileKernelKind::OneDnn,
                             onednn_flash::TileKernelKind::Reference);
    test_ws_matches_standard(onednn_flash::TileKernelKind::Reference,
                             onednn_flash::TileKernelKind::OneDnn);
    test_ws_matches_standard(onednn_flash::TileKernelKind::OneDnn,
                             onednn_flash::TileKernelKind::OneDnn);
    test_ws_matches_vector_api(onednn_flash::TileKernelKind::OneDnn,
                               onednn_flash::TileKernelKind::OneDnn);
    test_ws_matches_standard(onednn_flash::TileKernelKind::OneDnn,
                             onednn_flash::TileKernelKind::OneDnn,
                             onednn_flash::QkTileLayout::CopiedTransposed);
    test_ws_matches_vector_api(onednn_flash::TileKernelKind::OneDnn,
                               onednn_flash::TileKernelKind::OneDnn,
                               onednn_flash::QkTileLayout::CopiedTransposed);
#endif
    std::cout << "onednn_flash workspace attention tests passed\n";
    return 0;
}
