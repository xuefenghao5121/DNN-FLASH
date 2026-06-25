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

    const auto standard_out = standard();
    const auto tiled_out = tiled();
    const auto diff = flashone::max_abs_diff(standard_out, tiled_out);

    std::cout << "FlashOne benchmark (reference loops, not optimized backend)\n";
    std::cout << "shape: M=" << shape.query_tokens << " N=" << shape.key_tokens
              << " K=" << shape.head_dim << " D=" << shape.value_dim << " causal=1\n";
    std::cout << "max_abs_diff: " << diff << "\n";
    std::cout << "standard_attention_ms: " << time_ms(standard, repeat) << "\n";
    std::cout << "flash_attention_tiled_ms: " << time_ms(tiled, repeat) << "\n";
    return diff <= 1e-5f ? 0 : 1;
}
