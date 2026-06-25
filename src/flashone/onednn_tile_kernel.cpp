#include "flashone/onednn_tile_kernel.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <stdexcept>

namespace flashone {

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

    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream stream(engine);

    const auto m = static_cast<dnnl::memory::dim>(shape.m);
    const auto n = static_cast<dnnl::memory::dim>(shape.n);
    const auto k = static_cast<dnnl::memory::dim>(shape.k);

    auto a_md = dnnl::memory::desc({m, k}, dnnl::memory::data_type::f32, {k, 1});
    auto b_md = dnnl::memory::desc({k, n}, dnnl::memory::data_type::f32, {n, 1});
    auto c_md = dnnl::memory::desc({m, n}, dnnl::memory::data_type::f32, {n, 1});

    // oneDNN does not mutate src/weights; API takes non-const void* in this version.
    auto* a_ptr = const_cast<float*>(a.data());
    auto* b_ptr = const_cast<float*>(b.data());
    std::vector<float> c(shape.m * shape.n, 0.0f);

    dnnl::memory a_mem(a_md, engine, a_ptr);
    dnnl::memory b_mem(b_md, engine, b_ptr);
    dnnl::memory c_mem(c_md, engine, c.data());

    auto pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_md);
    dnnl::matmul primitive(pd);
    primitive.execute(stream, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem},
    });
    stream.wait();

    return c;
}

}  // namespace flashone
