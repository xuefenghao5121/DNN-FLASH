#include "flashone/onednn_brgemm_tile_kernel.hpp"

#ifdef FLASHONE_HAS_ONEDNN_BRGEMM

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ukernel.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace flashone {
namespace {

constexpr std::uintptr_t kScratchpadAlignment = 64;

std::uintptr_t align_up(std::uintptr_t value, std::uintptr_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

using dnnl::memory;
using dnnl::ukernel::brgemm;
using dnnl::ukernel::pack_type;
using dnnl::ukernel::transform;

struct BrgemmCacheKey {
    std::size_t m;
    std::size_t n;
    std::size_t k;
    std::size_t batch_size;
    std::size_t lda;
    std::size_t ldb;
    std::size_t ldc;
    std::size_t a_batch_stride;
    std::size_t b_batch_stride;

    bool operator<(const BrgemmCacheKey& other) const {
        return std::tie(m, n, k, batch_size, lda, ldb, ldc,
                        a_batch_stride, b_batch_stride) <
               std::tie(other.m, other.n, other.k, other.batch_size,
                        other.lda, other.ldb, other.ldc,
                        other.a_batch_stride, other.b_batch_stride);
    }
};

struct CachedBrgemmKernel {
    brgemm kernel;
    std::size_t scratchpad_size = 0;
    std::vector<std::pair<memory::dim, memory::dim>> offsets;
};

struct TransformCacheKey {
    std::size_t k;
    std::size_t n;
    std::size_t in_ld;
    std::size_t out_ld;

    bool operator<(const TransformCacheKey& other) const {
        return std::tie(k, n, in_ld, out_ld) <
               std::tie(other.k, other.n, other.in_ld, other.out_ld);
    }
};

struct CachedTransformKernel {
    transform kernel;
};

struct BrgemmContext {
    std::mutex mutex;
    std::map<BrgemmCacheKey, CachedBrgemmKernel> cache;
    std::map<TransformCacheKey, CachedTransformKernel> transform_cache;
};

BrgemmContext& context() {
    static BrgemmContext ctx;
    return ctx;
}

BrgemmCacheKey cache_key(const BrgemmShape& shape) {
    return BrgemmCacheKey{
        shape.m,
        shape.n,
        shape.k,
        shape.batch_size,
        shape.lda,
        shape.ldb,
        shape.ldc,
        shape.a_batch_stride,
        shape.b_batch_stride,
    };
}

bool transform_out_ld_supported(std::size_t out_ld) {
    return out_ld == 16 || out_ld == 32 || out_ld == 48 || out_ld == 64;
}

void materialize_transposed_b_reference(const float* b_transposed,
                                        float* packed_b,
                                        const MatmulShape& shape) {
    for (std::size_t kk = 0; kk < shape.k; ++kk) {
        for (std::size_t n = 0; n < shape.n; ++n) {
            packed_b[kk * shape.n + n] = b_transposed[n * shape.k + kk];
        }
    }
}

CachedTransformKernel make_transform_kernel(const TransformCacheKey& key) {
    if (key.k == 0 || key.n == 0 || key.in_ld == 0 || key.out_ld == 0) {
        throw std::invalid_argument("BRGEMM transform dimensions must be non-zero");
    }
    transform kernel(static_cast<memory::dim>(key.k),
                     static_cast<memory::dim>(key.n),
                     pack_type::trans,
                     static_cast<memory::dim>(key.in_ld),
                     static_cast<memory::dim>(key.out_ld),
                     memory::data_type::f32,
                     memory::data_type::f32,
                     /*allow_empty=*/true);
    if (!kernel) {
        throw std::runtime_error("failed to create oneDNN BRGEMM transposed-B transform ukernel");
    }
    kernel.generate();
    return CachedTransformKernel{std::move(kernel)};
}

CachedBrgemmKernel make_kernel(const BrgemmShape& shape) {
    if (shape.m == 0 || shape.n == 0 || shape.k == 0 || shape.batch_size == 0) {
        throw std::invalid_argument("BRGEMM dimensions and batch_size must be non-zero");
    }
    if (shape.lda < shape.k || shape.ldb < shape.n || shape.ldc < shape.n) {
        throw std::invalid_argument("BRGEMM leading dimensions are too small");
    }

    const auto pack = brgemm::get_B_pack_type(memory::data_type::f32,
                                              memory::data_type::f32);
    if (pack == pack_type::undef) {
        throw std::runtime_error("oneDNN BRGEMM ukernel does not support f32/f32 on this platform");
    }
    if (pack != pack_type::no_trans) {
        // Keep the first integration honest: FlashOne currently feeds row-major
        // B tiles. If the ukernel requires packed B, add transform support before
        // enabling this path for production.
        throw std::runtime_error("oneDNN BRGEMM ukernel requires packed B; transform path is not implemented yet");
    }

    brgemm kernel(static_cast<memory::dim>(shape.m),
                  static_cast<memory::dim>(shape.n),
                  static_cast<memory::dim>(shape.k),
                  static_cast<memory::dim>(shape.batch_size),
                  static_cast<memory::dim>(shape.lda),
                  static_cast<memory::dim>(shape.ldb),
                  static_cast<memory::dim>(shape.ldc),
                  memory::data_type::f32,
                  memory::data_type::f32,
                  memory::data_type::f32,
                  /*allow_empty=*/true);
    if (!kernel) {
        throw std::runtime_error("failed to create oneDNN BRGEMM ukernel");
    }
    kernel.set_add_C(false);
    if (!kernel.finalize()) {
        throw std::runtime_error("oneDNN BRGEMM ukernel finalize returned unsupported");
    }
    kernel.generate();

    CachedBrgemmKernel cached;
    cached.scratchpad_size = kernel.get_scratchpad_size();
    cached.offsets.reserve(shape.batch_size);
    const auto a_dt_size = memory::data_type_size(memory::data_type::f32);
    const auto b_dt_size = memory::data_type_size(memory::data_type::f32);
    for (std::size_t i = 0; i < shape.batch_size; ++i) {
        cached.offsets.emplace_back(
            static_cast<memory::dim>(i * shape.a_batch_stride * a_dt_size),
            static_cast<memory::dim>(i * shape.b_batch_stride * b_dt_size));
    }
    cached.kernel = std::move(kernel);
    return cached;
}

}  // namespace

bool onednn_brgemm_available() {
    return brgemm::get_B_pack_type(memory::data_type::f32,
                                   memory::data_type::f32) != pack_type::undef;
}

void* BrgemmScratchpad::data(std::size_t required_size) {
    if (required_size == 0) {
        return nullptr;
    }
    bytes.resize(required_size + kScratchpadAlignment - 1);
    const auto raw = reinterpret_cast<std::uintptr_t>(bytes.data());
    return reinterpret_cast<void*>(align_up(raw, kScratchpadAlignment));
}

void* BrgemmTransformWorkspace::data(std::size_t required_size) {
    if (required_size == 0) {
        return nullptr;
    }
    bytes.resize(required_size + kScratchpadAlignment - 1);
    const auto raw = reinterpret_cast<std::uintptr_t>(bytes.data());
    return reinterpret_cast<void*>(align_up(raw, kScratchpadAlignment));
}

void release_onednn_brgemm_hw_context() {
    brgemm::release_hw_context();
}

BrgemmKernelContext::~BrgemmKernelContext() {
    if (hw_context_active) {
        release_onednn_brgemm_hw_context();
    }
}

void matmul_tile_onednn_brgemm_inplace(const float* a,
                                       const float* b,
                                       float* c,
                                       const MatmulShape& shape) {
    thread_local BrgemmScratchpad scratchpad;
    matmul_tile_onednn_brgemm_inplace(a, b, c, shape, scratchpad, true);
}

void matmul_tile_onednn_brgemm_inplace(const float* a,
                                       const float* b,
                                       float* c,
                                       const MatmulShape& shape,
                                       BrgemmScratchpad& scratchpad) {
    matmul_tile_onednn_brgemm_inplace(a, b, c, shape, scratchpad, true);
}

void matmul_tile_onednn_brgemm_inplace(const float* a,
                                       const float* b,
                                       float* c,
                                       const MatmulShape& shape,
                                       BrgemmScratchpad& scratchpad,
                                       bool release_hw_context) {
    const std::size_t reduce_block = shape.k % 16 == 0 ? 16 : shape.k;
    const std::size_t batch_size = shape.k / reduce_block;
    BrgemmShape brg_shape{
        shape.m,
        shape.n,
        reduce_block,
        batch_size,
        shape.k,
        shape.n,
        shape.n,
        reduce_block,
        reduce_block * shape.n,
    };
    matmul_tile_onednn_brgemm_inplace(a, b, c, brg_shape, scratchpad, release_hw_context);
}

void matmul_tile_onednn_brgemm_inplace(const float* a,
                                       const float* b,
                                       float* c,
                                       const BrgemmShape& shape) {
    thread_local BrgemmScratchpad scratchpad;
    matmul_tile_onednn_brgemm_inplace(a, b, c, shape, scratchpad, true);
}

void matmul_tile_onednn_brgemm_inplace(const float* a,
                                       const float* b,
                                       float* c,
                                       const BrgemmShape& shape,
                                       BrgemmScratchpad& scratchpad) {
    matmul_tile_onednn_brgemm_inplace(a, b, c, shape, scratchpad, true);
}

void matmul_tile_onednn_brgemm_inplace(const float* a,
                                       const float* b,
                                       float* c,
                                       const BrgemmShape& shape,
                                       BrgemmScratchpad& scratchpad,
                                       bool release_hw_context) {
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument("BRGEMM pointers must be non-null");
    }

    auto& ctx = context();
    const auto key = cache_key(shape);
    CachedBrgemmKernel* cached = nullptr;
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        auto it = ctx.cache.find(key);
        if (it == ctx.cache.end()) {
            it = ctx.cache.emplace(key, make_kernel(shape)).first;
        }
        cached = &it->second;
    }

    void* scratchpad_ptr = scratchpad.data(cached->scratchpad_size);
    cached->kernel.set_hw_context();
    cached->kernel.execute(a, b, cached->offsets, c, scratchpad_ptr);
    if (release_hw_context) {
        brgemm::release_hw_context();
    }
}

void matmul_tile_onednn_brgemm_transposed_b_inplace(const float* a,
                                                    const float* b_transposed,
                                                    float* c,
                                                    const MatmulShape& shape,
                                                    BrgemmScratchpad& scratchpad,
                                                    BrgemmTransformWorkspace& transform_workspace,
                                                    bool release_hw_context) {
    if (a == nullptr || b_transposed == nullptr || c == nullptr) {
        throw std::invalid_argument("BRGEMM transposed-B pointers must be non-null");
    }
    if (shape.m == 0 || shape.n == 0 || shape.k == 0) {
        throw std::invalid_argument("BRGEMM transposed-B dimensions must be non-zero");
    }

    BrgemmShape brg_shape{
        shape.m,
        shape.n,
        shape.k,
        1,
        shape.k,
        shape.n,
        shape.n,
        shape.m * shape.k,
        shape.k * shape.n,
    };

    TransformCacheKey transform_key{
        shape.k,
        shape.n,
        shape.k,
        shape.n,
    };
    const bool use_onednn_transform = transform_out_ld_supported(shape.n);

    CachedTransformKernel* cached_transform = nullptr;
    CachedBrgemmKernel* cached_brgemm = nullptr;
    auto& ctx = context();
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        if (use_onednn_transform) {
            auto transform_it = ctx.transform_cache.find(transform_key);
            if (transform_it == ctx.transform_cache.end()) {
                transform_it = ctx.transform_cache.emplace(
                    transform_key, make_transform_kernel(transform_key)).first;
            }
            cached_transform = &transform_it->second;
        }

        const auto brgemm_key = cache_key(brg_shape);
        auto brgemm_it = ctx.cache.find(brgemm_key);
        if (brgemm_it == ctx.cache.end()) {
            brgemm_it = ctx.cache.emplace(brgemm_key, make_kernel(brg_shape)).first;
        }
        cached_brgemm = &brgemm_it->second;
    }

    // oneDNN's BRGEMM ukernel cannot consume an arbitrary strided B view.
    // Prefer the ukernel transform for supported out_ld values (documented as
    // 16/32/48/64); otherwise fall back to a small reference materialization so
    // partial K blocks and small tests still remain correct.
    void* packed_b = transform_workspace.data(shape.k * shape.n * sizeof(float));
    if (cached_transform != nullptr) {
        cached_transform->kernel.execute(b_transposed, packed_b);
    } else {
        materialize_transposed_b_reference(b_transposed, static_cast<float*>(packed_b), shape);
    }

    void* scratchpad_ptr = scratchpad.data(cached_brgemm->scratchpad_size);
    cached_brgemm->kernel.set_hw_context();
    cached_brgemm->kernel.execute(a, packed_b, cached_brgemm->offsets, c, scratchpad_ptr);
    if (release_hw_context) {
        brgemm::release_hw_context();
    }
}

void matmul_tile_onednn_brgemm_transposed_b_inplace(const float* a,
                                                    const float* b_transposed,
                                                    float* c,
                                                    const MatmulShape& shape,
                                                    BrgemmKernelContext& context,
                                                    bool release_hw_context) {
    matmul_tile_onednn_brgemm_transposed_b_inplace(a,
                                                   b_transposed,
                                                   c,
                                                   shape,
                                                   context.scratchpad,
                                                   context.transform_workspace,
                                                   release_hw_context);
    if (!release_hw_context) {
        context.hw_context_active = true;
    }
}

}  // namespace flashone

#endif  // FLASHONE_HAS_ONEDNN_BRGEMM
