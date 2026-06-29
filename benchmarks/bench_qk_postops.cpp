#include "onednn_flash/qk_score_tile_internal.hpp"

#ifdef ONEDNN_FLASH_HAS_ONEDNN
#include <oneapi/dnnl/dnnl.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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
    onednn_flash::ScoreModKind kind;
    bool has_scale;
    float scale;
    bool has_bias;
};

struct TimingStats {
    double mean_ms;
    double median_ms;
    double p90_ms;
    double min_ms;
    double max_ms;
    double stddev_ms;
};

struct Record {
    ShapeCase shape;
    std::string requested_backend;
    std::string actual_backend;
    std::string qk_layout;
    std::string score_mod;
    std::string sync_mode;
    std::string lowering_status;
    std::string fallback_reason;
    bool used_onednn_post_ops;
    float max_abs_diff;
    double time_ms;
    double time_ms_mean;
    double time_ms_median;
    double time_ms_p90;
    double time_ms_min;
    double time_ms_max;
    double time_ms_stddev;
    int warmup;
    int repeat;
    // Cache observability fields
    std::size_t cache_hits{0};
    std::size_t cache_misses{0};
    std::size_t handle_rebinds{0};
    std::size_t immediate_waits{0};
    std::size_t deferred_waits{0};
    std::size_t cache_size{0};
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

double median_of_sorted(const std::vector<double>& sorted_samples) {
    if (sorted_samples.empty()) {
        throw std::runtime_error("median requires at least one timing sample");
    }
    const std::size_t mid = sorted_samples.size() / 2;
    if (sorted_samples.size() % 2 == 1) {
        return sorted_samples[mid];
    }
    return (sorted_samples[mid - 1] + sorted_samples[mid]) * 0.5;
}

double nearest_rank_percentile(const std::vector<double>& sorted_samples, double percentile) {
    if (sorted_samples.empty()) {
        throw std::runtime_error("percentile requires at least one timing sample");
    }
    const auto rank = static_cast<std::size_t>(
        std::ceil(percentile * static_cast<double>(sorted_samples.size())));
    const auto index = std::min(sorted_samples.size() - 1, rank == 0 ? 0 : rank - 1);
    return sorted_samples[index];
}

TimingStats summarize_samples(const std::vector<double>& samples) {
    if (samples.empty()) {
        throw std::runtime_error("benchmark repeat must be positive");
    }

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(samples.size());
    double variance = 0.0;
    for (const double sample : samples) {
        const double delta = sample - mean;
        variance += delta * delta;
    }
    variance /= static_cast<double>(samples.size());

    return TimingStats{/*mean_ms=*/mean,
                       /*median_ms=*/median_of_sorted(sorted),
                       /*p90_ms=*/nearest_rank_percentile(sorted, 0.90),
                       /*min_ms=*/sorted.front(),
                       /*max_ms=*/sorted.back(),
                       /*stddev_ms=*/std::sqrt(variance)};
}

std::string git_commit_string() {
#ifdef ONEDNN_FLASH_GIT_COMMIT
    return ONEDNN_FLASH_GIT_COMMIT;
#else
    return "unknown";
#endif
}

std::string one_dnn_version_string() {
#ifdef ONEDNN_FLASH_HAS_ONEDNN
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

onednn_flash::StridedMatmulShape make_strided_shape(const ShapeCase& shape) {
    return onednn_flash::StridedMatmulShape{/*m=*/shape.m,
                                        /*n=*/shape.n,
                                        /*k=*/shape.d,
                                        /*a_stride_m=*/shape.d,
                                        /*a_stride_k=*/1,
                                        /*b_stride_k=*/1,
                                        /*b_stride_n=*/shape.d,
                                        /*c_stride_m=*/shape.n,
                                        /*c_stride_n=*/1};
}

onednn_flash::RuntimePlan make_plan(const ShapeCase& shape,
                                const ScoreCase& score_case,
                                bool force_reference) {
    onednn_flash::RuntimePlanInput input;
    input.batch = shape.batch;
    input.heads = shape.heads;
    input.query_length = static_cast<int>(shape.m);
    input.key_length = static_cast<int>(shape.n);
    input.head_dim = static_cast<int>(shape.d);
    input.value_dim = static_cast<int>(shape.dv);
    input.requested_score_mod = score_case.kind;
    input.has_scale = score_case.has_scale;
    input.scale_value = score_case.scale;
    input.requested_bias_kind = score_case.has_bias ? onednn_flash::BiasKind::SameShapeTile
                                                    : onednn_flash::BiasKind::None;
    input.force_reference = force_reference;
    input.enable_onednn = true;
#ifdef ONEDNN_FLASH_HAS_ONEDNN
    constexpr bool one_dnn_available = true;
#else
    constexpr bool one_dnn_available = false;
#endif
    return onednn_flash::make_runtime_plan(input,
                                       /*one_dnn_available=*/one_dnn_available,
                                       /*one_dnn_post_ops_available=*/one_dnn_available);
}

Record run_record(const ShapeCase& shape,
                  const ScoreCase& score_case,
                  const std::string& requested_backend,
                  const std::string& sync_mode,
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
    onednn_flash::QkScoreTileDebugInfo debug;
    onednn_flash::QkScoreTileExecuteOptions execute_options;
    const bool defer_wait = requested_backend == "onednn_matmul" && sync_mode == "defer_until_record_end";
    if (defer_wait) {
        execute_options.sync_mode = onednn_flash::QkScoreTileSyncMode::DeferUntilExplicitWait;
    }

    auto execute_once = [&]() {
        std::fill(actual.begin(), actual.end(), 0.0f);
        for (std::size_t tile = 0; tile < tile_count; ++tile) {
            onednn_flash::QkScoreTilePostOpsInput post_ops;
            post_ops.additive_bias = score_case.has_bias ? bias.data() + tile * shape.m * shape.n : nullptr;
            post_ops.additive_bias_stride_m = bias_stride_m;
            post_ops.additive_bias_stride_n = bias_stride_n;
            onednn_flash::qk_score_tile_inplace_with_options(q.data() + tile * shape.m * shape.d,
                                                         k.data() + tile * shape.n * shape.d,
                                                         actual.data() + tile * shape.m * shape.n,
                                                         strided_shape,
                                                         plan,
                                                         post_ops,
                                                         execute_options,
                                                         &debug);
        }
        if (defer_wait) {
            onednn_flash::qk_score_tile_wait_for_onednn();
        }
        volatile float sink = actual.empty() ? 0.0f : actual[0];
        (void)sink;
    };

    for (int i = 0; i < warmup; ++i) {
        execute_once();
    }

    onednn_flash::qk_score_tile_reset_cache_stats();

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<std::size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        const auto start = std::chrono::steady_clock::now();
        execute_once();
        const auto end = std::chrono::steady_clock::now();
        samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    const auto stats = summarize_samples(samples_ms);
    const auto cache_stats = onednn_flash::qk_score_tile_get_cache_stats();

    Record record;
    record.shape = shape;
    record.requested_backend = requested_backend;
    record.actual_backend = onednn_flash::to_string(debug.backend);
    record.qk_layout = onednn_flash::to_string(plan.qk_layout);
    record.score_mod = plan.score_mod_plan.signature;
    record.sync_mode = sync_mode;
    record.lowering_status = onednn_flash::to_string(debug.lowering_status);
    record.fallback_reason = onednn_flash::to_string(debug.fallback_reason);
    record.used_onednn_post_ops = debug.used_onednn_post_ops;
    record.max_abs_diff = max_abs_diff(expected, actual);
    record.time_ms = stats.mean_ms;
    record.time_ms_mean = stats.mean_ms;
    record.time_ms_median = stats.median_ms;
    record.time_ms_p90 = stats.p90_ms;
    record.time_ms_min = stats.min_ms;
    record.time_ms_max = stats.max_ms;
    record.time_ms_stddev = stats.stddev_ms;
    record.warmup = warmup;
    record.repeat = repeat;
    record.cache_hits = cache_stats.primitive_cache_hits;
    record.cache_misses = cache_stats.primitive_cache_misses;
    record.handle_rebinds = cache_stats.memory_handle_rebinds;
    record.immediate_waits = cache_stats.immediate_waits;
    record.deferred_waits = cache_stats.deferred_waits;
    record.cache_size = cache_stats.cache_size;
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
    out << "  \"schema\": \"onednn_flash.cpp_qk_postops_benchmark.v4\",\n";
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
        out << "      \"sync_mode\": \"" << r.sync_mode << "\",\n";
        out << "      \"lowering_status\": \"" << r.lowering_status << "\",\n";
        out << "      \"fallback_reason\": \"" << r.fallback_reason << "\",\n";
        out << "      \"used_onednn_post_ops\": " << (r.used_onednn_post_ops ? "true" : "false")
            << ",\n";
        out << "      \"max_abs_diff\": " << r.max_abs_diff << ",\n";
        out << "      \"time_ms\": " << r.time_ms << ",\n";
        out << "      \"time_ms_mean\": " << r.time_ms_mean << ",\n";
        out << "      \"time_ms_median\": " << r.time_ms_median << ",\n";
        out << "      \"time_ms_p90\": " << r.time_ms_p90 << ",\n";
        out << "      \"time_ms_min\": " << r.time_ms_min << ",\n";
        out << "      \"time_ms_max\": " << r.time_ms_max << ",\n";
        out << "      \"time_ms_stddev\": " << r.time_ms_stddev << ",\n";
        out << "      \"warmup\": " << r.warmup << ",\n";
        out << "      \"repeat\": " << r.repeat << ",\n";
        out << "      \"cache_stats\": {\n";
        out << "        \"primitive_cache_hits\": " << r.cache_hits << ",\n";
        out << "        \"primitive_cache_misses\": " << r.cache_misses << ",\n";
        out << "        \"memory_handle_rebinds\": " << r.handle_rebinds << ",\n";
        out << "        \"immediate_waits\": " << r.immediate_waits << ",\n";
        out << "        \"deferred_waits\": " << r.deferred_waits << ",\n";
        out << "        \"cache_size\": " << r.cache_size << "\n";
        out << "      }\n";
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
           "score_mod,sync_mode,lowering_status,fallback_reason,used_onednn_post_ops,max_abs_diff,"
           "time_ms,time_ms_mean,time_ms_median,time_ms_p90,time_ms_min,time_ms_max,"
           "time_ms_stddev,warmup,repeat,cache_hits,cache_misses,handle_rebinds,"
           "immediate_waits,deferred_waits,cache_size\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : records) {
        out << r.shape.batch << ',' << r.shape.heads << ',' << r.shape.m << ',' << r.shape.n
            << ',' << r.shape.d << ',' << r.shape.dv << ',' << r.requested_backend << ','
            << r.actual_backend << ',' << r.qk_layout << ',' << r.score_mod << ','
            << r.lowering_status << ',' << r.fallback_reason << ','
            << (r.used_onednn_post_ops ? "true" : "false") << ',' << r.max_abs_diff << ','
            << r.time_ms << ',' << r.time_ms_mean << ',' << r.time_ms_median << ','
            << r.time_ms_p90 << ',' << r.time_ms_min << ',' << r.time_ms_max << ','
            << r.time_ms_stddev << ',' << r.warmup << ',' << r.repeat << ','
            << r.cache_hits << ',' << r.cache_misses << ',' << r.handle_rebinds << ','
            << r.immediate_waits << ',' << r.deferred_waits << ',' << r.cache_size << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string json_path = "benchmarks/results/stage-1-postops/cpp-qk-postops.json";
    std::string csv_path = "benchmarks/results/stage-1-postops/cpp-qk-postops.csv";
    int warmup = 2;
    int repeat = 5;
    bool include_deferred_wait = false;

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
        } else if (arg == "--include-deferred-wait") {
            include_deferred_wait = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 2;
        }
    }

    if (warmup < 0) {
        std::cerr << "--warmup must be non-negative\n";
        return 2;
    }
    if (repeat <= 0) {
        std::cerr << "--repeat must be positive\n";
        return 2;
    }

    const std::vector<ShapeCase> shapes{{1, 1, 64, 64, 32, 32},
                                        {1, 1, 128, 128, 64, 64},
                                        {1, 4, 128, 128, 64, 64}};
    const std::vector<ScoreCase> score_cases{{"none", onednn_flash::ScoreModKind::None, false, 1.0f, false},
                                             {"scale", onednn_flash::ScoreModKind::Scale, true, 0.125f, false},
                                             {"scale_additive_bias",
                                              onednn_flash::ScoreModKind::ScaleAdditiveBias,
                                              true,
                                              0.125f,
                                              true}};
    const std::vector<std::string> backends{"reference", "onednn_matmul"};

    std::vector<Record> records;
    for (const auto& shape : shapes) {
        for (const auto& score_case : score_cases) {
            for (const auto& backend : backends) {
                records.push_back(run_record(shape,
                                             score_case,
                                             backend,
                                             "wait_after_execute",
                                             warmup,
                                             repeat));
                if (include_deferred_wait && backend == "onednn_matmul") {
                    records.push_back(run_record(shape,
                                                 score_case,
                                                 backend,
                                                 "defer_until_record_end",
                                                 warmup,
                                                 repeat));
                }
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
