#pragma once

#include "flashone/tile_kernel.hpp"

namespace flashone {

std::vector<float> matmul_tile_onednn(const std::vector<float>& a,
                                      const std::vector<float>& b,
                                      const MatmulShape& shape);

}  // namespace flashone
