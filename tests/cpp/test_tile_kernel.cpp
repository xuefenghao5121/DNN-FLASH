#include "flashone/tile_kernel.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> make_values(std::size_t size, float offset) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.19f + offset) * 0.4f;
    }
    return values;
}

void require_close(const std::vector<float>& a,
                   const std::vector<float>& b,
                   float tolerance,
                   const char* message) {
    if (a.size() != b.size()) {
        throw std::runtime_error("size mismatch");
    }
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    if (max_diff > tolerance) {
        std::cerr << message << " max_diff=" << max_diff << "\n";
        throw std::runtime_error(message);
    }
}

void test_reference_dispatch() {
    const flashone::MatmulShape shape{5, 7, 3};
    const auto a = make_values(shape.m * shape.k, 0.2f);
    const auto b = make_values(shape.k * shape.n, 1.2f);
    const auto direct = flashone::matmul_tile_reference(a, b, shape);
    const auto dispatched = flashone::matmul_tile(flashone::TileKernelKind::Reference, a, b, shape);
    require_close(direct, dispatched, 1e-6f, "reference dispatch mismatch");
}

#ifdef FLASHONE_HAS_ONEDNN
void test_onednn_matches_reference() {
    const flashone::MatmulShape shape{8, 9, 6};
    const auto a = make_values(shape.m * shape.k, 0.4f);
    const auto b = make_values(shape.k * shape.n, 1.4f);
    const auto expected = flashone::matmul_tile_reference(a, b, shape);
    const auto actual = flashone::matmul_tile(flashone::TileKernelKind::OneDnn, a, b, shape);
    require_close(expected, actual, 1e-5f, "oneDNN tile kernel mismatch");
}
#endif

}  // namespace

int main() {
    test_reference_dispatch();
#ifdef FLASHONE_HAS_ONEDNN
    test_onednn_matches_reference();
#endif
    std::cout << "flashone tile kernel tests passed\n";
    return 0;
}
