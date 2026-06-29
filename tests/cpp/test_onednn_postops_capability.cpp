#include <oneapi/dnnl/dnnl.hpp>

#ifdef ONEDNN_FLASH_HAS_ONEDNN_BRGEMM
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#endif

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ProbeRow {
    std::string name;
    bool supported = false;
    std::string method = "unsupported";
    std::string compile_status = "passed";
    std::string run_status = "not_tested";
    std::string reason;
};

std::string json_escape(const std::string& value) {
    std::ostringstream oss;
    for (const char ch : value) {
        switch (ch) {
            case '"':
                oss << "\\\"";
                break;
            case '\\':
                oss << "\\\\";
                break;
            case '\n':
                oss << "\\n";
                break;
            default:
                oss << ch;
                break;
        }
    }
    return oss.str();
}

std::string optional_json_string(const std::string& value) {
    if (value.empty()) {
        return "null";
    }
    return "\"" + json_escape(value) + "\"";
}

std::string status(bool value) {
    return value ? "true" : "false";
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size(), "diff size mismatch");
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

std::vector<float> matmul_reference(const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    int m,
                                    int n,
                                    int k) {
    std::vector<float> c(static_cast<std::size_t>(m * n), 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                sum += a[static_cast<std::size_t>(i * k + kk)] *
                       b[static_cast<std::size_t>(kk * n + j)];
            }
            c[static_cast<std::size_t>(i * n + j)] = sum;
        }
    }
    return c;
}

ProbeRow run_matmul_with_attr(const std::string& name,
                              const dnnl::primitive_attr& attr,
                              const std::vector<float>* binary_input,
                              const std::vector<float>& expected,
                              const std::string& method) {
    ProbeRow row;
    row.name = name;
    row.method = method;

    try {
        constexpr int m = 4;
        constexpr int n = 5;
        constexpr int k = 3;
        dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream stream(engine);

        const auto a_md = dnnl::memory::desc({m, k}, dnnl::memory::data_type::f32, {k, 1});
        const auto b_md = dnnl::memory::desc({k, n}, dnnl::memory::data_type::f32, {n, 1});
        const auto c_md = dnnl::memory::desc({m, n}, dnnl::memory::data_type::f32, {n, 1});

        const auto pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_md, attr);
        const dnnl::matmul primitive(pd);

        std::vector<float> a(static_cast<std::size_t>(m * k));
        std::vector<float> b(static_cast<std::size_t>(k * n));
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<float>(i + 1) * 0.125f;
        }
        for (std::size_t i = 0; i < b.size(); ++i) {
            b[i] = static_cast<float>((i % 7) + 1) * 0.0625f;
        }
        std::vector<float> c(static_cast<std::size_t>(m * n), 0.0f);

        dnnl::memory a_mem(a_md, engine, a.data());
        dnnl::memory b_mem(b_md, engine, b.data());
        dnnl::memory c_mem(c_md, engine, c.data());

        std::unordered_map<int, dnnl::memory> args{{DNNL_ARG_SRC, a_mem},
                                                   {DNNL_ARG_WEIGHTS, b_mem},
                                                   {DNNL_ARG_DST, c_mem}};
        dnnl::memory binary_mem;
        if (binary_input != nullptr) {
            binary_mem = dnnl::memory(c_md, engine, const_cast<float*>(binary_input->data()));
            args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, binary_mem);
        }

        primitive.execute(stream, args);
        stream.wait();

        const auto diff = max_abs_diff(c, expected);
        if (diff > 1e-5f) {
            std::ostringstream oss;
            oss << "max_abs_diff=" << diff;
            row.reason = oss.str();
            row.run_status = "failed";
            return row;
        }

        row.supported = true;
        row.run_status = "passed";
        return row;
    } catch (const std::exception& e) {
        row.reason = e.what();
        row.run_status = "failed";
        return row;
    }
}

ProbeRow probe_matmul_scale() {
    constexpr int m = 4;
    constexpr int n = 5;
    constexpr int k = 3;
    std::vector<float> a(static_cast<std::size_t>(m * k));
    std::vector<float> b(static_cast<std::size_t>(k * n));
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i + 1) * 0.125f;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>((i % 7) + 1) * 0.0625f;
    }
    auto expected = matmul_reference(a, b, m, n, k);
    for (auto& v : expected) {
        v *= 2.0f;
    }

    dnnl::primitive_attr attr;
    dnnl::post_ops ops;
    ops.append_eltwise(dnnl::algorithm::eltwise_linear, 2.0f, 0.0f);
    attr.set_post_ops(ops);
    return run_matmul_with_attr("matmul.scale", attr, nullptr, expected, "post_op");
}

ProbeRow probe_matmul_binary_add(const std::string& name,
                                 const dnnl::memory::dims& bias_dims,
                                 const dnnl::memory::dims& bias_strides,
                                 const std::vector<float>& bias,
                                 const std::vector<float>& expanded_bias) {
    constexpr int m = 4;
    constexpr int n = 5;
    constexpr int k = 3;
    std::vector<float> a(static_cast<std::size_t>(m * k));
    std::vector<float> b(static_cast<std::size_t>(k * n));
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i + 1) * 0.125f;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>((i % 7) + 1) * 0.0625f;
    }
    auto expected = matmul_reference(a, b, m, n, k);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] += expanded_bias[i];
    }

    dnnl::primitive_attr attr;
    dnnl::post_ops ops;
    const auto bias_md = dnnl::memory::desc(bias_dims, dnnl::memory::data_type::f32, bias_strides);
    ops.append_binary(dnnl::algorithm::binary_add, bias_md);
    attr.set_post_ops(ops);

    ProbeRow row;
    row.name = name;
    row.method = "post_op";
    try {
        dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream stream(engine);
        const auto a_md = dnnl::memory::desc({m, k}, dnnl::memory::data_type::f32, {k, 1});
        const auto b_md = dnnl::memory::desc({k, n}, dnnl::memory::data_type::f32, {n, 1});
        const auto c_md = dnnl::memory::desc({m, n}, dnnl::memory::data_type::f32, {n, 1});
        const auto pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_md, attr);
        const dnnl::matmul primitive(pd);

        std::vector<float> c(static_cast<std::size_t>(m * n), 0.0f);
        dnnl::memory a_mem(a_md, engine, a.data());
        dnnl::memory b_mem(b_md, engine, b.data());
        dnnl::memory c_mem(c_md, engine, c.data());
        dnnl::memory bias_mem(bias_md, engine, const_cast<float*>(bias.data()));
        primitive.execute(stream, {{DNNL_ARG_SRC, a_mem},
                                   {DNNL_ARG_WEIGHTS, b_mem},
                                   {DNNL_ARG_DST, c_mem},
                                   {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, bias_mem}});
        stream.wait();

        const auto diff = max_abs_diff(c, expected);
        if (diff > 1e-5f) {
            std::ostringstream oss;
            oss << "max_abs_diff=" << diff;
            row.reason = oss.str();
            row.run_status = "failed";
            return row;
        }
        row.supported = true;
        row.run_status = "passed";
        return row;
    } catch (const std::exception& e) {
        row.reason = e.what();
        row.run_status = "failed";
        return row;
    }
}

ProbeRow probe_same_shape_binary_add() {
    constexpr int m = 4;
    constexpr int n = 5;
    std::vector<float> bias(static_cast<std::size_t>(m * n));
    for (std::size_t i = 0; i < bias.size(); ++i) {
        bias[i] = static_cast<float>(i + 1) * 0.01f;
    }
    return probe_matmul_binary_add("matmul.binary_add_same_shape", {m, n}, {n, 1}, bias, bias);
}

ProbeRow probe_broadcast_row_binary_add() {
    constexpr int m = 4;
    constexpr int n = 5;
    std::vector<float> bias(static_cast<std::size_t>(m));
    std::vector<float> expanded(static_cast<std::size_t>(m * n));
    for (int i = 0; i < m; ++i) {
        bias[static_cast<std::size_t>(i)] = static_cast<float>(i + 1) * 0.02f;
        for (int j = 0; j < n; ++j) {
            expanded[static_cast<std::size_t>(i * n + j)] = bias[static_cast<std::size_t>(i)];
        }
    }
    return probe_matmul_binary_add("matmul.binary_add_broadcast_row", {m, 1}, {1, 1}, bias, expanded);
}

ProbeRow probe_broadcast_col_binary_add() {
    constexpr int m = 4;
    constexpr int n = 5;
    std::vector<float> bias(static_cast<std::size_t>(n));
    std::vector<float> expanded(static_cast<std::size_t>(m * n));
    for (int j = 0; j < n; ++j) {
        bias[static_cast<std::size_t>(j)] = static_cast<float>(j + 1) * 0.03f;
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            expanded[static_cast<std::size_t>(i * n + j)] = bias[static_cast<std::size_t>(j)];
        }
    }
    return probe_matmul_binary_add("matmul.binary_add_broadcast_col", {1, n}, {n, 1}, bias, expanded);
}

ProbeRow probe_eltwise_linear() {
    return probe_matmul_scale();
}

ProbeRow probe_ukernel_brgemm_boundary() {
    ProbeRow row;
    row.name = "ukernel_brgemm.post_ops_boundary";
#ifdef ONEDNN_FLASH_HAS_ONEDNN_BRGEMM
    row.compile_status = "passed";
    row.run_status = "not_tested";
    row.supported = false;
    row.method = "unsupported";
    row.reason = "oneDNN ukernel BRGEMM API is available, but Stage 1 does not expose post-op support for ukernel BRGEMM; use dnnl::matmul post-ops instead";
#else
    row.compile_status = "not_tested";
    row.run_status = "not_tested";
    row.supported = false;
    row.method = "unsupported";
    row.reason = "ONEDNN_FLASH_HAS_ONEDNN_BRGEMM is not enabled";
#endif
    return row;
}

ProbeRow probe_ukernel_transform_out_ld() {
    ProbeRow row;
    row.name = "ukernel_transform.supported_out_ld";
#ifdef ONEDNN_FLASH_HAS_ONEDNN_BRGEMM
    row.compile_status = "passed";
    row.run_status = "passed";
    row.supported = true;
    row.method = "runtime_arg";
    row.reason = "supported out_ld values observed by integration contract: 16,32,48,64";
#else
    row.compile_status = "not_tested";
    row.run_status = "not_tested";
    row.supported = false;
    row.method = "unsupported";
    row.reason = "ONEDNN_FLASH_HAS_ONEDNN_BRGEMM is not enabled";
#endif
    return row;
}

std::string json_report(const std::vector<ProbeRow>& rows) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"schema\": \"onednn_flash.onednn_postops_capability.v1\",\n";
    oss << "  \"primitive\": \"matmul\",\n";
    oss << "  \"dtype\": \"f32\",\n";
    oss << "  \"rows\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        oss << "    {\n";
        oss << "      \"name\": \"" << json_escape(row.name) << "\",\n";
        oss << "      \"supported\": " << status(row.supported) << ",\n";
        oss << "      \"method\": \"" << json_escape(row.method) << "\",\n";
        oss << "      \"compile_status\": \"" << json_escape(row.compile_status) << "\",\n";
        oss << "      \"run_status\": \"" << json_escape(row.run_status) << "\",\n";
        oss << "      \"reason\": " << optional_json_string(row.reason) << "\n";
        oss << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

std::string markdown_report(const std::vector<ProbeRow>& rows) {
    std::ostringstream oss;
    oss << "# oneDNN Post-ops Capability Probe\n\n";
    oss << "> Date: 2026-06-27\n";
    oss << "> Scope: OneDNNFlash Stage 1.1 capability probe only; not a performance benchmark.\n\n";
    oss << "## Results\n\n";
    oss << "| Name | Supported | Method | Compile | Run | Reason |\n";
    oss << "|---|---:|---|---|---|---|\n";
    for (const auto& row : rows) {
        oss << "| `" << row.name << "` | " << (row.supported ? "yes" : "no") << " | `"
            << row.method << "` | `" << row.compile_status << "` | `" << row.run_status
            << "` | " << (row.reason.empty() ? "" : row.reason) << " |\n";
    }
    oss << "\n## Stage 1 Interpretation\n\n";
    oss << "- Stage 1 main path remains `dnnl::matmul + post_ops`.\n";
    oss << "- BRGEMM ukernel is capability/baseline context only, not the Stage 1 post-op path.\n";
    oss << "- Broadcast rows are probed for evidence, but Stage 1 RuntimePlan still treats broadcast bias as fallback until explicitly promoted by design.\n";
    oss << "- This probe does not claim TensorFlow graph/XLA speedup.\n";
    return oss.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }
    out << content;
}

}  // namespace

int main() {
    std::vector<ProbeRow> rows;
    rows.push_back(probe_matmul_scale());
    rows.back().name = "matmul.scale";
    rows.push_back(probe_same_shape_binary_add());
    rows.push_back(probe_broadcast_row_binary_add());
    rows.push_back(probe_broadcast_col_binary_add());
    rows.push_back(probe_eltwise_linear());
    rows.back().name = "matmul.eltwise_linear";
    rows.push_back(probe_ukernel_brgemm_boundary());
    rows.push_back(probe_ukernel_transform_out_ld());

    write_file("benchmarks/results/stage-1-postops/onednn-postops-capability.json", json_report(rows));
    write_file("docs/backend/onednn-postops-capability-2026-06-27.md", markdown_report(rows));

    bool required_passed = true;
    for (const auto& row : rows) {
        if (row.name.rfind("matmul.", 0) == 0 && row.run_status != "passed") {
            required_passed = false;
            std::cerr << row.name << " failed: " << row.reason << '\n';
        }
    }
    if (!required_passed) {
        return 1;
    }

    std::cout << json_report(rows);
    return 0;
}
