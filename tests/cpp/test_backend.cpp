#include "onednn_flash/backend.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> make_values(std::size_t size, float offset) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::cos(static_cast<float>(i) * 0.11f + offset) * 0.3f;
    }
    return values;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_backend_dispatch() {
    const onednn_flash::AttentionShape shape{5, 6, 4, 3};
    const auto q = make_values(shape.query_tokens * shape.head_dim, 0.1f);
    const auto k = make_values(shape.key_tokens * shape.head_dim, 1.1f);
    const auto v = make_values(shape.key_tokens * shape.value_dim, 2.1f);
    const onednn_flash::AttentionOptions options{1.0f / std::sqrt(static_cast<float>(shape.head_dim)),
                                             true,
                                             2};
    const auto standard = onednn_flash::run_attention(
        onednn_flash::AttentionBackendKind::StandardReference, q, k, v, shape, options);
    const auto tiled = onednn_flash::run_attention(
        onednn_flash::AttentionBackendKind::FlashTiledReference, q, k, v, shape, options);
    require(onednn_flash::max_abs_diff(standard, tiled) < 1e-5f, "backend dispatch mismatch");
}

void test_block_mask_and_bias() {
    const onednn_flash::AttentionShape shape{4, 4, 3, 2};
    const auto q = make_values(shape.query_tokens * shape.head_dim, 0.2f);
    const auto k = make_values(shape.key_tokens * shape.head_dim, 1.2f);
    const auto v = make_values(shape.key_tokens * shape.value_dim, 2.2f);

    const onednn_flash::BlockMask mask{
        2,
        2,
        2,
        2,
        {
            1, 0,
            0, 1,
        },
    };

    onednn_flash::AttentionOptions options;
    options.scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    options.causal = false;
    options.key_block_size = 2;
    options.block_mask = &mask;
    options.score_bias = [](std::size_t qi, std::size_t kj) {
        return static_cast<float>(qi) * 0.01f - static_cast<float>(kj) * 0.02f;
    };

    const auto standard = onednn_flash::standard_attention(q, k, v, shape, options);
    const auto tiled = onednn_flash::flash_attention_tiled(q, k, v, shape, options);
    require(onednn_flash::max_abs_diff(standard, tiled) < 1e-5f, "block mask + bias mismatch");
}

}  // namespace

int main() {
    test_backend_dispatch();
    test_block_mask_and_bias();
    std::cout << "onednn_flash backend tests passed\n";
    return 0;
}
