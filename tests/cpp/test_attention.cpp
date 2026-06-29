#include "onednn_flash/attention.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> make_values(std::size_t size, float offset) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.17f + offset) * 0.5f;
    }
    return values;
}

void require_close(float diff, float tolerance, const char* name) {
    if (diff > tolerance) {
        std::cerr << name << " diff " << diff << " > " << tolerance << '\n';
        throw std::runtime_error("attention outputs differ");
    }
}

void test_case(bool causal, std::size_t block_size) {
    const onednn_flash::AttentionShape shape{9, 11, 7, 5};
    const auto q = make_values(shape.query_tokens * shape.head_dim, 0.1f);
    const auto k = make_values(shape.key_tokens * shape.head_dim, 1.3f);
    const auto v = make_values(shape.key_tokens * shape.value_dim, 2.7f);
    const onednn_flash::AttentionOptions options{1.0f / std::sqrt(static_cast<float>(shape.head_dim)),
                                             causal,
                                             block_size};
    const auto expected = onednn_flash::standard_attention(q, k, v, shape, options);
    const auto actual = onednn_flash::flash_attention_tiled(q, k, v, shape, options);
    require_close(onednn_flash::max_abs_diff(expected, actual), 1e-5f, causal ? "causal" : "dense");
}

}  // namespace

int main() {
    test_case(false, 1);
    test_case(false, 3);
    test_case(false, 64);
    test_case(true, 1);
    test_case(true, 4);
    test_case(true, 64);
    std::cout << "onednn_flash C++ attention tests passed\n";
    return 0;
}
