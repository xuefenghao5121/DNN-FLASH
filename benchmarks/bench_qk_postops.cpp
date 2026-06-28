#include "flashone/qk_score_tile_internal.hpp"

#ifdef FLASHONE_HAS_ONEDNN
#include <oneapi/dnnl/dnnl.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ShapeCase {
    int batch;
    int heads;
    std::size_t m;
    std::size_t n;
    std::size_t d;
    std::size_t dv;
};

struct ScoreCase {
    const char* name;
    flashone::ScoreModKind kind;
    bool has_scale;
    float scale;
    bool has_bias;
};

struct Record {
    ShapeCase shape;
    std::string requested_backend;
    std::string actual_backend;
    std::string qk_layout;
    std::string score_mod;
    std::string lowering_status;
    std::string fallback_reason;
    bool used_onednn_post_ops;
    float max_abs_diff;
    double time_ms;
    int warmup;
    int repeat;
};

std::vector<float> make_values(std::size_t size, float offset) {
    std::vector<float> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.017f + offset) * 0.25f;
    }
    return values;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("max_abs_diff size mismatch");
    }
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

std::string git_commit_string() {
#ifdef FLASHONE_GIT_COMMIT
    return FLASHONE_GIT_COMMIT;
#else
    return "unknown";
#endif
}

std::string one_dnn_version_string() {
#ifdef FLASHONE_HAS_ONEDNN
    const auto* version = dnnl_version();
    if (version == nullptr) {
        return "unknown";
    }
    std::ostringstream oss;
    oss << version->major << '.' << version->minor << '.' << version->patch;
    return oss.str();
#else
    return "not_built";
#endif
}

flashone::StridedMatmulShape make_strided_shape(const ShapeCase& shape) {
    return flashone::StridedMatmulShape{/*m=*/shape.m,
                                        /*n=*/shape.n,
                                        /*k=*/shape.d,
                                        /*a_stride_m=*/shape.d,
                                        /*a_stride_k=*/1,
                                        /*b_stride_k=*/1,
                                        /*b_stride_n=*/shape.d,
                                        /*c_stride_m=*/shape.n,
                                        /*c_stride_n=*/1};
}

flashone::RuntimePlan make_plan(const ShapeCase& shape,
                                const ScoreCase& score_case,
                                bool force_reference) {
    flashone::RuntimePlanInput input;
    input.batch = shape.batch;
    input.heads = shape.heads;
    input.query_length = static_cast<int>(shape.m);
    input.key_length = static_cast<int>(shape.n);
    input.head_dim = static_cast<int>(shape.d);
    input.value_dim = static_cast<int>(shape.dv);
    input.requested_score_mod = score_case.kind;
    input.has_scale = score_case.has_scale;
    input.scale_value = score_case.scale;
    input.requested_bias_kind = score_case.has_bias ? flashone::BiasKind::SameShapeTile
                                                    : flashone::BiasKind::None;
    input.force_reference = force_reference;
    input.enable_onednn = true;
#ifdef FLASHONE_HAS_ONEDNN
    constexpr bool one_dnn_available = true;
#else
    constexpr bool one_dnn_available = false;
#endif
    return flashone::make_runtime_plan(input,
                                       /*one_dnn_available=*/one_dnn_available,
                                       /*one_dnn_post_ops_available=*/one_dnn_available);
}

Record run_record(const ShapeCase& shape,
                  const ScoreCase& score_case,
                  const std::string& requested_backend,
                  int warmup,
                  int repeat) {
    const bool force_reference = requested_backend == "reference";
    const auto plan = make_plan(shape, score_case, force_reference);
    const std::size_t tile_count = static_cast<std::size_t>(shape.batch) *
                                   static_cast<std::size_t>(shape.heads);
    const auto q = make_values(tile_count * shape.m * shape.d, 0.1f);
    const auto k = make_values(tile_count * shape.n * shape.d, 0.7f);
    const auto bias = make_values(tile_count * shape.m * shape.n, 1.3f);
    const std::size_t bias_stride_m = shape.n;
    const std::size_t bias_stride_n = 1;

    std::vector<float> expected(tile_count * shape.m * shape.n, 0.0f);
    for (std::size_t tile = 0; tile < tile_count; ++tile) {
        const float* q_tile = q.data() + tile * shape.m * shape.d;
        const float* k_tile = k.data() + tile * shape.n * shape.d;
        const float* bias_tile = score_case.has_bias ? bias.data() + tile * shape.m * shape.n : nullptr;
        float* expected_tile = expected.data() + tile * shape.m * shape.n;
        for (std::size_t i = 0; i < shape.m; ++i) {
            for (std::size_t j = 0; j < shape.n; ++j) {
                float value = 0.0f;
                for (std::size_t kk = 0; kk < shape.d; ++kk) {
                    value += q_tile[i * shape.d + kk] * k_tile[j * shape.d + kk];
                }
                if (score_case.has_scale) {
                    value *= score_case.scale;
                }
                if (bias_tile != nullptr) {
                    value += bias_tile[i * bias_stride_m + j * bias_stride_n];
                }
                expected_tile[i * shape.n + j] = value;
            }
        }
    }

    std::vector<float> actual(tile_count * shape.m * shape.n, 0.0f);
    const auto strided_shape = make_strided_shape(shape);
    flashone::QkScoreTileDebugInfo debug;

    auto execute_once = [&]() {
        std::fill(actual.begin(), actual.end(), 0.0f);
        for (std::size_t tile = 0; tile < tile_count; ++tile) {
            flashone::QkScoreTilePostOpsInput post_ops;
            post_ops.additive_bias = score_case.has_bias ? bias.data() + tile * shape.m * shape.n : nullptr;
            post_ops.additive_bias_stride_m = bias_stride_m;
            post_ops.additive_bias_stride_n = bias_stride_n;
            flashone::qk_score_tile_inplace(q.data() + tile * shape.m * shape.d,
                                            k.data() + tile * shape.n * shape.d,
                                            actual.data() + tile * shape.m * shape.n,
                                            strided_shape,
                                            plan,
                                            post_ops,
                                            &debug);
        }
        volatile float sink = actual.empty() ? 0.0f : actual[0];
        (void)sink;
    };

    for (int i = 0; i < warmup; ++i) {
        execute_once();
    }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        execute_once();
    }
    const auto end = std::chrono::steady_clock::now();
    const double time_ms = std::chrono::duration<double, std::milli>(end - start).count() /
                           static_cast<double>(repeat);

    Record record;
    record.shape = shape;
    record.requested_backend = requested_backend;
    record.actual_backend = flashone::to_string(debug.backend);
    record.qk_layout = flashone::to_string(plan.qk_layout);
    record.score_mod = plan.score_mod_plan.signature;
    record.lowering_status = flashone::to_string(debug.lowering_status);
    record.fallback_reason = flashone::to_string(debug.fallback_reason);
    record.used_onednn_post_ops = debug.used_onednn_post_ops;
    record.max_abs_diff = max_abs_diff(expected, actual);
    record.time_ms = time_ms;
    record.warmup = warmup;
    record.repeat = repeat;
    return record;
}

void ensure_parent_dir(const std::string& path) {
    const std::filesystem::path fs_path(path);
    if (fs_path.has_parent_path()) {
        std::filesystem::create_directories(fs_path.parent_path());
    }
}

void write_json(const std::string& path, const std::vector<Record>& records) {
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open JSON output: " + path);
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"schema\": \"flashone.cpp_qk_postops_benchmark.v1\",\n";
    out << "  \"benchmark_level\": \"cpp_qk_score_tile\",\n";
    out << "  \"git_commit\": \"" << git_commit_string() << "\",\n";
    out << "  \"one_dnn_version\": \"" << one_dnn_version_string() << "\",\n";
    out << "  \"interpretation_boundary\": \"C++ QK score tile benchmark only; does not claim TensorFlow graph/XLA speedup.\",\n";
    out << "  \"records\": [\n";
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        out << "    {\n";
        out << "      \"shape\": {\"batch\": " << r.shape.batch << ", \"heads\": "
            << r.shape.heads << ", \"m\": " << r.shape.m << ", \"n\": " << r.shape.n
            << ", \"head_dim\": " << r.shape.d << ", \"value_dim\": " << r.shape.dv
            << "},\n";
        out << "      \"requested_backend\": \"" << r.requested_backend << "\",\n";
        out << "      \"actual_backend\": \"" << r.actual_backend << "\",\n";
        out << "      \"qk_layout\": \"" << r.qk_layout << "\",\n";
        out << "      \"score_mod\": \"" << r.score_mod << "\",\n";
        out << "      \"lowering_status\": \"" << r.lowering_status << "\",\n";
        out << "      \"fallback_reason\": \"" << r.fallback_reason << "\",\n";
        out << "      \"used_onednn_post_ops\": " << (r.used_onednn_post_ops ? "true" : "false")
            << ",\n";
        out << "      \"max_abs_diff\": " << r.max_abs_diff << ",\n";
        out << "      \"time_ms\": " << r.time_ms << ",\n";
        out << "      \"warmup\": " << r.warmup << ",\n";
        out << "      \"repeat\": " << r.repeat << "\n";
        out << "    }" << (i + 1 == records.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

void write_csv(const std::string& path, const std::vector<Record>& records) {
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open CSV output: " + path);
    }
    out << "batch,heads,m,n,head_dim,value_dim,requested_backend,actual_backend,qk_layout,"
           "score_mod,lowering_status,fallback_reason,used_onednn_post_ops,max_abs_diff,"
           "time_ms,warmup,repeat\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : records) {
        out << r.shape.batch << ',' << r.shape.heads << ',' << r.shape.m << ',' << r.shape.n
            << ',' << r.shape.d << ',' << r.shape.dv << ',' << r.requested_backend << ','
            << r.actual_backend << ',' << r.qk_layout << ',' << r.score_mod << ','
            << r.lowering_status << ',' << r.fallback_reason << ','
            << (r.used_onednn_post_ops ? "true" : "false") << ',' << r.max_abs_diff << ','
            << r.time_ms << ',' << r.warmup << ',' << r.repeat << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string json_path = "benchmarks/results/stage-1-postops/cpp-qk-postops.json";
    std::string csv_path = "benchmarks/results/stage-1-postops/cpp-qk-postops.csv";
    int warmup = 2;
    int repeat = 5;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--output-json" && i + 1 < argc) {
            json_path = argv[++i];
        } else if (arg == "--output-csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 2;
        }
    }

    const std::vector<ShapeCase> shapes{{1, 1, 64, 64, 32, 32},
                                        {1, 1, 128, 128, 64, 64},
                                        {1, 4, 128, 128, 64, 64}};
    const std::vector<ScoreCase> score_cases{{"none", flashone::ScoreModKind::None, false, 1.0f, false},
                                             {"scale", flashone::ScoreModKind::Scale, true, 0.125f, false},
                                             {"scale_additive_bias",
                                              flashone::ScoreModKind::ScaleAdditiveBias,
                                              true,
                                              0.125f,
                                              true}};
    const std::vector<std::string> backends{"reference", "onednn_matmul"};

    std::vector<Record> records;
    for (const auto& shape : shapes) {
        for (const auto& score_case : score_cases) {
            for (const auto& backend : backends) {
                records.push_back(run_record(shape, score_case, backend, warmup, repeat));
            }
        }
    }

    write_json(json_path, records);
    write_csv(csv_path, records);
    std::cout << "wrote " << records.size() << " QK post-op benchmark records\n";
    std::cout << "json: " << json_path << "\n";
    std::cout << "csv: " << csv_path << "\n";
    return 0;
}
