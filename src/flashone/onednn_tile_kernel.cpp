#include "flashone/onednn_tile_kernel.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

namespace flashone {
namespace {

struct CachedMatmulPrimitive {
    dnnl::memory::desc a_md;
    dnnl::memory::desc b_md;
    dnnl::memory::desc c_md;
    dnnl::matmul primitive;
};

struct OneDnnTileContext {
    dnnl::engine engine{dnnl::engine::kind::cpu, 0};
    dnnl::stream stream{engine};
    std::mutex mutex;
    std::map<std::tuple<std::size_t, std::size_t, std::size_t>, CachedMatmulPrimitive> cache;
};

OneDnnTileContext& context() {
    static OneDnnTileContext ctx;
    return ctx;
}

CachedMatmulPrimitive make_cached_primitive(dnnl::engine& engine, const MatmulShape& shape) {
    const auto m = static_cast<dnnl::memory::dim>(shape.m);
    const auto n = static_cast<dnnl::memory::dim>(shape.n);
    const auto k = static_cast<dnnl::memory::dim>(shape.k);

    auto a_md = dnnl::memory::desc({m, k}, dnnl::memory::data_type::f32, {k, 1});
    auto b_md = dnnl::memory::desc({k, n}, dnnl::memory::data_type::f32, {n, 1});
    auto c_md = dnnl::memory::desc({m, n}, dnnl::memory::data_type::f32, {n, 1});
    auto pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_md);
    return CachedMatmulPrimitive{a_md, b_md, c_md, dnnl::matmul(pd)};
}

}  // namespace

std::vector<float> matmul_tile_onednn(const std::vector<float>& a,
                                      const std::vector<float>& b,
                                      const MatmulShape& shape) {
    if (shape.m == 0 || shape.n == 0 || shape.k == 0) {
        throw std::invalid_argument("matmul dimensions must be non-zero");
    }
    if (a.size() != shape.m * shape.k) {
        throw std::invalid_argument("A size does not match matmul shape");
    }
    if (b.size() != shape.k * shape.n) {
        throw std::invalid_argument("B size does not match matmul shape");
    }

    std::vector<float> c(shape.m * shape.n, 0.0f);
    matmul_tile_onednn_inplace(a.data(), b.data(), c.data(), shape);
    return c;
}

void matmul_tile_onednn_inplace(const float* a,
                                 const float* b,
                                 float* c,
                                 const MatmulShape& shape) {
    if (shape.m == 0 || shape.n == 0 || shape.k == 0) {
        throw std::invalid_argument("matmul dimensions must be non-zero");
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument("matmul pointers must be non-null");
    }

    auto& ctx = context();
    const auto key = std::make_tuple(shape.m, shape.n, shape.k);

    std::lock_guard<std::mutex> lock(ctx.mutex);
    auto it = ctx.cache.find(key);
    if (it == ctx.cache.end()) {
        it = ctx.cache.emplace(key, make_cached_primitive(ctx.engine, shape)).first;
    }
    auto& cached = it->second;

    auto* a_ptr = const_cast<float*>(a);
    auto* b_ptr = const_cast<float*>(b);

    dnnl::memory a_mem(cached.a_md, ctx.engine, a_ptr);
    dnnl::memory b_mem(cached.b_md, ctx.engine, b_ptr);
    dnnl::memory c_mem(cached.c_md, ctx.engine, c);

    cached.primitive.execute(ctx.stream, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem},
    });
    ctx.stream.wait();
}

}  // namespace flashone
