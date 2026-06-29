#include "onednn_flash/backend.hpp"

#include <stdexcept>

namespace onednn_flash {

const char* backend_name(AttentionBackendKind kind) {
    switch (kind) {
        case AttentionBackendKind::StandardReference:
            return "standard_reference";
        case AttentionBackendKind::FlashTiledReference:
            return "flash_tiled_reference";
    }
    return "unknown";
}

std::vector<float> run_attention(AttentionBackendKind kind,
                                 const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v,
                                 const AttentionShape& shape,
                                 const AttentionOptions& options) {
    switch (kind) {
        case AttentionBackendKind::StandardReference:
            return standard_attention(q, k, v, shape, options);
        case AttentionBackendKind::FlashTiledReference:
            return flash_attention_tiled(q, k, v, shape, options);
    }
    throw std::invalid_argument("Unknown attention backend kind");
}

}  // namespace onednn_flash
