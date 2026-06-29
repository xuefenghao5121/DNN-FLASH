#include "onednn_flash/attention.hpp"

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
    const onednn_flash::AttentionShape shape{128, 128, 64, 64};
    const auto q = make_values(shape.query_tokens * shape.head_dim, 0.1f);
    const auto k = make_values(shape.key_tokens * shape.head_dim, 1.3f);
    const auto v = make_values(shape.key_tokens * shape.value_dim, 2.7f);
    const onednn_flash::AttentionOptions options{1.0f / std::sqrt(static_cast<float>(shape.head_dim)),
                                             true,
                                             32};
    constexpr int repeat = 5;

    const auto standard = [&]() { return onednn_flash::standard_attention(q, k, v, shape, options); };
    const auto tiled = [&]() { return onednn_flash::flash_attention_tiled(q, k, v, shape, options); };

    auto q_tile_options = options;
    q_tile_options.query_block_size = 16;
    q_tile_options.qk_tile_kernel = onednn_flash::TileKernelKind::Reference;
    const auto q_tile = [&]() { return onednn_flash::flash_attention_q_tile(q, k, v, shape, q_tile_options); };

    auto qk_pv_options = q_tile_options;
    qk_pv_options.pv_tile_kernel = onednn_flash::TileKernelKind::Reference;
    const auto qk_pv_tile = [&]() {
        return onednn_flash::flash_attention_qk_pv_tile(q, k, v, shape, qk_pv_options);
    };

#ifdef ONEDNN_FLASH_HAS_ONEDNN
    auto q_tile_onednn_options = q_tile_options;
    q_tile_onednn_options.qk_tile_kernel = onednn_flash::TileKernelKind::OneDnn;
    const auto q_tile_onednn = [&]() {
        return onednn_flash::flash_attention_q_tile(q, k, v, shape, q_tile_onednn_options);
    };

    auto qk_pv_onednn_options = q_tile_onednn_options;
    qk_pv_onednn_options.pv_tile_kernel = onednn_flash::TileKernelKind::OneDnn;
    qk_pv_onednn_options.qk_tile_layout = onednn_flash::QkTileLayout::StridedK;
    const auto qk_pv_onednn = [&]() {
        return onednn_flash::flash_attention_qk_pv_tile(q, k, v, shape, qk_pv_onednn_options);
    };

    auto qk_pv_onednn_copied_options = qk_pv_onednn_options;
    qk_pv_onednn_copied_options.qk_tile_layout = onednn_flash::QkTileLayout::CopiedTransposed;
    const auto qk_pv_onednn_copied = [&]() {
        return onednn_flash::flash_attention_qk_pv_tile(q, k, v, shape, qk_pv_onednn_copied_options);
    };
#endif

#ifdef ONEDNN_FLASH_HAS_ONEDNN_BRGEMM
    auto qk_pv_brgemm_options = qk_pv_onednn_copied_options;
    qk_pv_brgemm_options.qk_tile_kernel = onednn_flash::TileKernelKind::OneDnnBrgemm;
    qk_pv_brgemm_options.pv_tile_kernel = onednn_flash::TileKernelKind::OneDnnBrgemm;
    const auto qk_pv_brgemm = [&]() {
        return onednn_flash::flash_attention_qk_pv_tile(q, k, v, shape, qk_pv_brgemm_options);
    };

    auto qk_pv_brgemm_transformed_options = qk_pv_brgemm_options;
    qk_pv_brgemm_transformed_options.qk_tile_layout = onednn_flash::QkTileLayout::BrgemmTransformedK;
    const auto qk_pv_brgemm_transformed = [&]() {
        return onednn_flash::flash_attention_qk_pv_tile(q, k, v, shape, qk_pv_brgemm_transformed_options);
    };
#endif

    const auto standard_out = standard();
    const auto tiled_out = tiled();
    const auto q_tile_out = q_tile();
    const auto qk_pv_tile_out = qk_pv_tile();
    const auto diff = onednn_flash::max_abs_diff(standard_out, tiled_out);
    const auto q_tile_diff = onednn_flash::max_abs_diff(standard_out, q_tile_out);
    const auto qk_pv_tile_diff = onednn_flash::max_abs_diff(standard_out, qk_pv_tile_out);

#ifdef ONEDNN_FLASH_HAS_ONEDNN
    const auto q_tile_onednn_out = q_tile_onednn();
    const auto qk_pv_onednn_out = qk_pv_onednn();
    const auto qk_pv_onednn_copied_out = qk_pv_onednn_copied();
    const auto q_tile_onednn_diff = onednn_flash::max_abs_diff(standard_out, q_tile_onednn_out);
    const auto qk_pv_onednn_diff = onednn_flash::max_abs_diff(standard_out, qk_pv_onednn_out);
    const auto qk_pv_onednn_copied_diff =
        onednn_flash::max_abs_diff(standard_out, qk_pv_onednn_copied_out);
#endif
#ifdef ONEDNN_FLASH_HAS_ONEDNN_BRGEMM
    const auto qk_pv_brgemm_out = qk_pv_brgemm();
    const auto qk_pv_brgemm_diff = onednn_flash::max_abs_diff(standard_out, qk_pv_brgemm_out);
    const auto qk_pv_brgemm_transformed_out = qk_pv_brgemm_transformed();
    const auto qk_pv_brgemm_transformed_diff =
        onednn_flash::max_abs_diff(standard_out, qk_pv_brgemm_transformed_out);
#endif

    std::cout << "OneDNNFlash benchmark (row-tiled reference + QK/PV backend variants)\n";
    std::cout << "shape: M=" << shape.query_tokens << " N=" << shape.key_tokens
              << " K=" << shape.head_dim << " D=" << shape.value_dim << " causal=1\n";
    std::cout << "max_abs_diff_row_tiled: " << diff << "\n";
    std::cout << "max_abs_diff_q_tile: " << q_tile_diff << "\n";
    std::cout << "max_abs_diff_qk_pv_tile: " << qk_pv_tile_diff << "\n";
#ifdef ONEDNN_FLASH_HAS_ONEDNN
    std::cout << "max_abs_diff_q_tile_onednn: " << q_tile_onednn_diff << "\n";
    std::cout << "max_abs_diff_qk_pv_onednn: " << qk_pv_onednn_diff << "\n";
    std::cout << "max_abs_diff_qk_pv_onednn_copied_k: " << qk_pv_onednn_copied_diff << "\n";
#endif
#ifdef ONEDNN_FLASH_HAS_ONEDNN_BRGEMM
    std::cout << "max_abs_diff_qk_pv_onednn_brgemm: " << qk_pv_brgemm_diff << "\n";
    std::cout << "max_abs_diff_qk_pv_onednn_brgemm_transformed_k: "
              << qk_pv_brgemm_transformed_diff << "\n";
#endif

    std::cout << "standard_attention_ms: " << time_ms(standard, repeat) << "\n";
    std::cout << "flash_attention_tiled_ms: " << time_ms(tiled, repeat) << "\n";
    std::cout << "flash_attention_q_tile_ref_ms: " << time_ms(q_tile, repeat) << "\n";
    std::cout << "flash_attention_qk_pv_tile_ref_ms: " << time_ms(qk_pv_tile, repeat) << "\n";
#ifdef ONEDNN_FLASH_HAS_ONEDNN
    std::cout << "flash_attention_q_tile_onednn_ms: " << time_ms(q_tile_onednn, repeat) << "\n";
    std::cout << "flash_attention_qk_pv_onednn_ms: " << time_ms(qk_pv_onednn, repeat) << "\n";
    std::cout << "flash_attention_qk_pv_onednn_copied_k_ms: "
              << time_ms(qk_pv_onednn_copied, repeat) << "\n";
#endif
#ifdef ONEDNN_FLASH_HAS_ONEDNN_BRGEMM
    std::cout << "flash_attention_qk_pv_onednn_brgemm_ms: "
              << time_ms(qk_pv_brgemm, repeat) << "\n";
    std::cout << "flash_attention_qk_pv_onednn_brgemm_transformed_k_ms: "
              << time_ms(qk_pv_brgemm_transformed, repeat) << "\n";
#endif

    bool ok = diff <= 1e-5f && q_tile_diff <= 1e-5f && qk_pv_tile_diff <= 1e-5f;
#ifdef ONEDNN_FLASH_HAS_ONEDNN
    ok = ok && q_tile_onednn_diff <= 1e-5f && qk_pv_onednn_diff <= 1e-5f &&
         qk_pv_onednn_copied_diff <= 1e-5f;
#endif
#ifdef ONEDNN_FLASH_HAS_ONEDNN_BRGEMM
    ok = ok && qk_pv_brgemm_diff <= 1e-5f && qk_pv_brgemm_transformed_diff <= 1e-5f;
#endif
    return ok ? 0 : 1;
}
