#include <oneapi/dnnl/dnnl.hpp>

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream stream(engine);

    constexpr int m = 2;
    constexpr int k = 3;
    constexpr int n = 2;

    std::vector<float> a = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    std::vector<float> b = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f,
    };
    std::vector<float> c(m * n, 0.0f);

    auto a_md = dnnl::memory::desc({m, k}, dnnl::memory::data_type::f32, {k, 1});
    auto b_md = dnnl::memory::desc({k, n}, dnnl::memory::data_type::f32, {n, 1});
    auto c_md = dnnl::memory::desc({m, n}, dnnl::memory::data_type::f32, {n, 1});

    dnnl::memory a_mem(a_md, engine, a.data());
    dnnl::memory b_mem(b_md, engine, b.data());
    dnnl::memory c_mem(c_md, engine, c.data());

    auto pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_md);
    dnnl::matmul matmul(pd);

    matmul.execute(stream, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem},
    });
    stream.wait();

    const std::vector<float> expected = {
        58.0f, 64.0f,
        139.0f, 154.0f,
    };
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < c.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(c[i] - expected[i]));
    }

    std::cout << "oneDNN smoke matmul max_diff=" << max_diff << "\n";
    std::cout << "oneDNN runtime version=" << dnnl_version()->major << "."
              << dnnl_version()->minor << "." << dnnl_version()->patch << "\n";
    return max_diff < 1e-6f ? 0 : 1;
}
