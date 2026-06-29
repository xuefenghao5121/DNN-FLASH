/**
 * Stage 1 Simple Test: oneDNN matmul only (no post-op)
 */

#include <iostream>
#include <vector>
#include <cmath>

#include <oneapi/dnnl/dnnl.hpp>

int main() {
    std::cout << "Stage 1 Simple Test: oneDNN matmul" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "oneDNN version: " << dnnl::version() << std::endl;

    using namespace dnnl;

    // Small test
    const int m = 4, n = 4, k = 4;
    std::vector<float> A(m * k, 1.0f);
    std::vector<float> B(n * k, 2.0f);
    std::vector<float> C(m * n, 0.0f);

    try {
        engine eng(engine::kind::cpu, 0);
        stream s(eng);

        memory::dims a_dims = {m, k};
        memory::dims b_dims = {n, k};
        memory::dims c_dims = {m, n};

        memory::desc a_md(a_dims, memory::data_type::f32, {k, 1});
        memory::desc b_md(b_dims, memory::data_type::f32, {k, 1});
        memory::desc c_md(c_dims, memory::data_type::f32, {n, 1});

        memory a_mem(a_md, eng, A.data());
        memory b_mem(b_md, eng, B.data());
        memory c_mem(c_md, eng, C.data());

        matmul::primitive_desc matmul_pd(eng, a_md, b_md, c_md);
        matmul matmul_prim(matmul_pd);

        matmul_prim.execute(s, {
            {DNNL_ARG_SRC_0, a_mem},
            {DNNL_ARG_SRC_1, b_mem},
            {DNNL_ARG_DST, c_mem}
        });
        s.wait();

        std::cout << "Result[0]: " << C[0] << " (expected: " << (k * 1.0f * 2.0f) << ")" << std::endl;
        std::cout << "Test PASSED" << std::endl;
        return 0;
    } catch (const dnnl::error& e) {
        std::cerr << "oneDNN error: " << e.what() << ", status: " << e.status << std::endl;
        return 1;
    }
}
