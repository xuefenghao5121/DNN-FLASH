#pragma once

#include "onednn_flash/attention.hpp"

namespace onednn_flash {

enum class AttentionBackendKind {
    StandardReference,
    FlashTiledReference,
};

const char* backend_name(AttentionBackendKind kind);

std::vector<float> run_attention(AttentionBackendKind kind,
                                 const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v,
                                 const AttentionShape& shape,
                                 const AttentionOptions& options);

}  // namespace onednn_flash
