#include "flashone/attention.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<float> make_values(std::size_t size, float offset) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.013f + offset) * 0.25f;
    }
    return values;
}

template <typename Fn>
double time_ms(Fn&& fn, int repeat) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        const auto out = fn();
        volatile float sink = out.empty() ? 0.0f : out[0];
        (void)sink;
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / repeat;
}

}  // namespace

int main() {
    const flashone::AttentionShape shape{128, 128, 64, 64};
    const auto q = make_values(shape.query_tokens * shape.head_dim, 0.1f);
    const auto k = make_values(shape.key_tokens * shape.head_dim, 1.3f);
    const auto v = make_values(shape.key_tokens * shape.value_dim, 2.7f);
    const flashone::AttentionOptions options{1.0f / std::sqrt(static_cast<float>(shape.head_dim)),
                                             true,
                                             32};
    constexpr int repeat = 5;

    const auto standard = [&]() { return flashone::standard_attention(q, k, v, shape, options); };
    const auto tiled = [&]() { return flashone::flash_attention_tiled(q, k, v, shape, options); };

    auto q_tile_options = options;
    q_tile_options.query_block_size = 16;
    q_tile_options.qk_tile_kernel = flashone::TileKernelKind::Reference;
    const auto q_tile = [&]() { return flashone::flash_attention_q_tile(q, k, v, shape, q_tile_options); };

    auto qk_pv_options = q_tile_options;
    qk_pv_options.pv_tile_kernel = flashone::TileKernelKind::Reference;
    const auto qk_pv_tile = [&]() {
        return flashone::flash_attention_qk_pv_tile(q, k, v, shape, qk_pv_options);
    };

#ifdef FLASHONE_HAS_ONEDNN
    auto q_tile_onednn_options = q_tile_options;
    q_tile_onednn_options.qk_tile_kernel = flashone::TileKernelKind::OneDnn;
    const auto q_tile_onednn = [&]() {
        return flashone::flash_attention_q_tile(q, k, v, shape, q_tile_onednn_options);
    };

    auto qk_pv_onednn_options = q_tile_onednn_options;
    qk_pv_onednn_options.pv_tile_kernel = flashone::TileKernelKind::OneDnn;
    const auto qk_pv_onednn = [&]() {
        return flashone::flash_attention_qk_pv_tile(q, k, v, shape, qk_pv_onednn_options);
    };
#endif

    const auto standard_out = standard();
    const auto tiled_out = tiled();
    const auto q_tile_out = q_tile();
    const auto qk_pv_tile_out = qk_pv_tile();
    const auto diff = flashone::max_abs_diff(standard_out, tiled_out);
    const auto q_tile_diff = flashone::max_abs_diff(standard_out, q_tile_out);
    const auto qk_pv_tile_diff = flashone::max_abs_diff(standard_out, qk_pv_tile_out);
#ifdef FLASHONE_HAS_ONEDNN
    const auto q_tile_onednn_out = q_tile_onednn();
    const auto qk_pv_onednn_out = qk_pv_onednn();
    const auto q_tile_onednn_diff = flashone::max_abs_diff(standard_out, q_tile_onednn_out);
    const auto qk_pv_onednn_diff = flashone::max_abs_diff(standard_out, qk_pv_onednn_out);
#endif

    std::cout << "FlashOne benchmark (row-tiled reference + QK tile backend variants)\n";
    std::cout << "shape: M=" << shape.query_tokens << " N=" << shape.key_tokens
              << " K=" << shape.head_dim << " D=" << shape.value_dim << " causal=1\n";
    std::cout << "max_abs_diff_row_tiled: " << diff << "\n";
    std::cout << "max_abs_diff_q_tile: " << q_tile_diff << "\n";
    std::cout << "max_abs_diff_qk_pv_tile: " << qk_pv_tile_diff << "\n";
#ifdef FLASHONE_HAS_ONEDNN
    std::cout << "max_abs_diff_q_tile_onednn: " << q_tile_onednn_diff << "\n";
    std::cout << "max_abs_diff_qk_pv_onednn: " << qk_pv_onednn_diff << "\n";
#endif
    std::cout << "standard_attention_ms: " << time_ms(standard, repeat) << "\n";
    std::cout << "flash_attention_tiled_ms: " << time_ms(tiled, repeat) << "\n";
    std::cout << "flash_attention_q_tile_ref_ms: " << time_ms(q_tile, repeat) << "\n";
    std::cout << "flash_attention_qk_pv_tile_ref_ms: " << time_ms(qk_pv_tile, repeat) << "\n";
#ifdef FLASHONE_HAS_ONEDNN
    std::cout << "flash_attention_q_tile_onednn_ms: " << time_ms(q_tile_onednn, repeat) << "\n";
    std::cout << "flash_attention_qk_pv_onednn_ms: " << time_ms(qk_pv_onednn, repeat) << "\n";
#endif
    bool ok = diff <= 1e-5f && q_tile_diff <= 1e-5f && qk_pv_tile_diff <= 1e-5f;
#ifdef FLASHONE_HAS_ONEDNN
    ok = ok && q_tile_onednn_diff <= 1e-5f && qk_pv_onednn_diff <= 1e-5f;
#endif
    return ok ? 0 : 1;
}
