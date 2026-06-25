#pragma once

#include "flashone/attention.hpp"

namespace flashone {

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

}  // namespace flashone
