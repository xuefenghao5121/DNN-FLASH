#pragma once

#include "onednn_flash/tile_kernel.hpp"

namespace onednn_flash {

std::vector<float> matmul_tile_onednn(const std::vector<float>& a,
                                      const std::vector<float>& b,
                                      const MatmulShape& shape);

// Inplace variant: writes directly to c, no vector allocation.
void matmul_tile_onednn_inplace(const float* a,
                                 const float* b,
                                 float* c,
                                 const MatmulShape& shape);

}  // namespace onednn_flash
