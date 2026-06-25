#include "flashone/attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace flashone {
namespace {

void validate_inputs(const std::vector<float>& q,
                     const std::vector<float>& k,
                     const std::vector<float>& v,
                     const AttentionShape& shape) {
    const auto expected_q = shape.query_tokens * shape.head_dim;
    const auto expected_k = shape.key_tokens * shape.head_dim;
    const auto expected_v = shape.key_tokens * shape.value_dim;
    if (q.size() != expected_q) {
        throw std::invalid_argument("Q size does not match shape");
    }
    if (k.size() != expected_k) {
        throw std::invalid_argument("K size does not match shape");
    }
    if (v.size() != expected_v) {
        throw std::invalid_argument("V size does not match shape");
    }
    if (shape.query_tokens == 0 || shape.key_tokens == 0 || shape.head_dim == 0 ||
        shape.value_dim == 0) {
        throw std::invalid_argument("Attention dimensions must be non-zero");
    }
}

float dot_row(const std::vector<float>& a,
              std::size_t a_row,
              const std::vector<float>& b,
              std::size_t b_row,
              std::size_t width) {
    float sum = 0.0f;
    const auto a_base = a_row * width;
    const auto b_base = b_row * width;
    for (std::size_t d = 0; d < width; ++d) {
        sum += a[a_base + d] * b[b_base + d];
    }
    return sum;
}

bool masked_out(std::size_t query_index, std::size_t key_index, const AttentionOptions& options) {
    if (options.causal && key_index > query_index) {
        return true;
    }
    if (options.block_mask != nullptr && !options.block_mask->allows(query_index, key_index)) {
        return true;
    }
    return false;
}

float score_bias(std::size_t query_index, std::size_t key_index, const AttentionOptions& options) {
    return options.score_bias ? options.score_bias(query_index, key_index) : 0.0f;
}

}  // namespace

bool BlockMask::allows(std::size_t query_index, std::size_t key_index) const {
    if (query_block_size == 0 || key_block_size == 0) {
        throw std::invalid_argument("BlockMask block sizes must be non-zero");
    }
    if (query_blocks == 0 || key_blocks == 0) {
        throw std::invalid_argument("BlockMask dimensions must be non-zero");
    }
    if (allowed.size() != query_blocks * key_blocks) {
        throw std::invalid_argument("BlockMask allowed bitmap size does not match dimensions");
    }
    const auto qb = query_index / query_block_size;
    const auto kb = key_index / key_block_size;
    if (qb >= query_blocks || kb >= key_blocks) {
        return false;
    }
    return allowed[qb * key_blocks + kb] != 0;
}

std::vector<float> standard_attention(const std::vector<float>& q,
                                      const std::vector<float>& k,
                                      const std::vector<float>& v,
                                      const AttentionShape& shape,
                                      const AttentionOptions& options) {
    validate_inputs(q, k, v, shape);

    std::vector<float> output(shape.query_tokens * shape.value_dim, 0.0f);
    std::vector<float> scores(shape.key_tokens, 0.0f);

    for (std::size_t qi = 0; qi < shape.query_tokens; ++qi) {
        float row_max = -std::numeric_limits<float>::infinity();
        bool has_valid_key = false;
        for (std::size_t kj = 0; kj < shape.key_tokens; ++kj) {
            if (masked_out(qi, kj, options)) {
                scores[kj] = -std::numeric_limits<float>::infinity();
                continue;
            }
            scores[kj] = dot_row(q, qi, k, kj, shape.head_dim) * options.scale +
                         score_bias(qi, kj, options);
            row_max = std::max(row_max, scores[kj]);
            has_valid_key = true;
        }
        if (!has_valid_key) {
            continue;
        }

        float denom = 0.0f;
        for (std::size_t kj = 0; kj < shape.key_tokens; ++kj) {
            if (std::isinf(scores[kj]) && scores[kj] < 0.0f) {
                continue;
            }
            scores[kj] = std::exp(scores[kj] - row_max);
            denom += scores[kj];
        }

        for (std::size_t vd = 0; vd < shape.value_dim; ++vd) {
            float acc = 0.0f;
            for (std::size_t kj = 0; kj < shape.key_tokens; ++kj) {
                if (std::isinf(scores[kj]) && scores[kj] < 0.0f) {
                    continue;
                }
                acc += scores[kj] * v[kj * shape.value_dim + vd];
            }
            output[qi * shape.value_dim + vd] = acc / denom;
        }
    }

    return output;
}

std::vector<float> flash_attention_tiled(const std::vector<float>& q,
                                         const std::vector<float>& k,
                                         const std::vector<float>& v,
                                         const AttentionShape& shape,
                                         const AttentionOptions& options) {
    validate_inputs(q, k, v, shape);
    if (options.key_block_size == 0) {
        throw std::invalid_argument("key_block_size must be non-zero");
    }

    std::vector<float> output(shape.query_tokens * shape.value_dim, 0.0f);

    for (std::size_t qi = 0; qi < shape.query_tokens; ++qi) {
        float running_max = -std::numeric_limits<float>::infinity();
        float running_sum = 0.0f;
        std::vector<float> acc(shape.value_dim, 0.0f);

        for (std::size_t block_begin = 0; block_begin < shape.key_tokens;
             block_begin += options.key_block_size) {
            const auto block_end = std::min(shape.key_tokens, block_begin + options.key_block_size);

            if (options.causal && block_begin > qi) {
                break;
            }

            float block_max = -std::numeric_limits<float>::infinity();
            bool has_valid_key = false;
            std::vector<float> block_scores(block_end - block_begin, 0.0f);

            for (std::size_t kj = block_begin; kj < block_end; ++kj) {
                if (masked_out(qi, kj, options)) {
                    block_scores[kj - block_begin] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const float score = dot_row(q, qi, k, kj, shape.head_dim) * options.scale +
                                    score_bias(qi, kj, options);
                block_scores[kj - block_begin] = score;
                block_max = std::max(block_max, score);
                has_valid_key = true;
            }

            if (!has_valid_key) {
                continue;
            }

            const float new_max = std::max(running_max, block_max);
            const float old_scale = std::isinf(running_max) && running_max < 0.0f
                                        ? 0.0f
                                        : std::exp(running_max - new_max);

            for (float& value : acc) {
                value *= old_scale;
            }
            running_sum *= old_scale;

            float block_sum = 0.0f;
            for (std::size_t local = 0; local < block_scores.size(); ++local) {
                const float score = block_scores[local];
                if (std::isinf(score) && score < 0.0f) {
                    continue;
                }
                const float weight = std::exp(score - new_max);
                const auto kj = block_begin + local;
                block_sum += weight;
                for (std::size_t vd = 0; vd < shape.value_dim; ++vd) {
                    acc[vd] += weight * v[kj * shape.value_dim + vd];
                }
            }

            running_sum += block_sum;
            running_max = new_max;
        }

        if (running_sum > 0.0f) {
            for (std::size_t vd = 0; vd < shape.value_dim; ++vd) {
                output[qi * shape.value_dim + vd] = acc[vd] / running_sum;
            }
        }
    }

    return output;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Cannot compare vectors with different sizes");
    }
    float diff = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff;
}

}  // namespace flashone
