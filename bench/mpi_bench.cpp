//  mpi_bench - distributed benchmark driver for CUDAsearch
//
//  Usage:
//    mpirun -np 4 ./mpi_bench [options]
//
//  Options:
//    --data <dir>         Path to dataset directory
//    --dataset <name>     Dataset name (default: sift1m)
//    --kernel <name>      Kernel: cpu, naive, tiled, int8, int8_tiled
//    --k <int>            Number of neighbours (default: 10)
//    --batch <int>        Query batch size (default: 100)
//    --n <int>            Max database size (-1 = full, default: -1)
//    --trials <int>       Number of timed runs (default: 5)
//    --synthetic          Use synthetic random data instead of disk
//    --syn-N <int>        Synthetic base size (default: 100000)
//    --syn-B <int>        Synthetic query count (default: 1000)
//    --syn-d <int>        Synthetic dimension (default: 128)
//    --csv                Print result as CSV row

#include "core/cpu_search.hpp"
#include "io/fvecs_loader.hpp"
#include "mpi/distributed_search.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
    std::string data_dir;
    std::string dataset = "sift1m";
#ifdef HAVE_CUDA
    std::string kernel = "tiled";
#else
    std::string kernel = "cpu";
#endif
    int k = 10;
    int batch = 100;
    int n_limit = -1;
    int trials = 5;
    bool use_synthetic = false;
    int syn_N = 100'000;
    int syn_B = 1000;
    int syn_d = 128;
    bool csv = false;
};

Config parse_args(int argc, char* argv[])
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* s) { return std::strcmp(argv[i], s) == 0; };
        auto next_int = [&]() { return std::atoi(argv[++i]); };
        auto next_str = [&]() -> std::string { return argv[++i]; };

        if (is("--data")) cfg.data_dir = next_str();
        else if (is("--dataset")) cfg.dataset = next_str();
        else if (is("--kernel")) cfg.kernel = next_str();
        else if (is("--k")) cfg.k = next_int();
        else if (is("--batch")) cfg.batch = next_int();
        else if (is("--n")) cfg.n_limit = next_int();
        else if (is("--trials")) cfg.trials = next_int();
        else if (is("--synthetic")) cfg.use_synthetic = true;
        else if (is("--syn-N")) cfg.syn_N = next_int();
        else if (is("--syn-B")) cfg.syn_B = next_int();
        else if (is("--syn-d")) cfg.syn_d = next_int();
        else if (is("--csv")) cfg.csv = true;
        else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            std::exit(1);
        }
    }
    return cfg;
}

struct Stats {
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
};

Stats summarize(const std::vector<double>& values)
{
    Stats s;
    s.min = *std::min_element(values.begin(), values.end());
    s.max = *std::max_element(values.begin(), values.end());
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    return s;
}

void push_timing(std::vector<double>& totals,
                 std::vector<double>& bcasts,
                 std::vector<double>& locals,
                 std::vector<double>& gathers,
                 std::vector<double>& merges,
                 const core::DistributedSearchTiming& t)
{
    totals.push_back(t.total_ms);
    bcasts.push_back(t.query_bcast_ms);
    locals.push_back(t.local_search_ms);
    gathers.push_back(t.candidate_gather_ms);
    merges.push_back(t.merge_ms);
}

void l2_normalize(std::vector<float>& data, int rows, int d)
{
    for (int i = 0; i < rows; ++i) {
        float* row = data.data() + static_cast<size_t>(i) * d;
        float norm = 0.0f;
        for (int j = 0; j < d; ++j) {
            norm += row[j] * row[j];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-9f) {
            for (int j = 0; j < d; ++j) {
                row[j] /= norm;
            }
        }
    }
}

io::Dataset make_synthetic_benchmark_dataset(int N, int B, int d)
{
    io::Dataset ds;
    ds.name = "random";
    ds.n_base = N;
    ds.n_queries = B;
    ds.dim = d;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    ds.base.resize(static_cast<size_t>(N) * d);
    ds.queries.resize(static_cast<size_t>(B) * d);
    for (float& value : ds.base) {
        value = dist(rng);
    }
    for (float& value : ds.queries) {
        value = dist(rng);
    }

    l2_normalize(ds.base, N, d);
    l2_normalize(ds.queries, B, d);
    return ds;
}

} // namespace

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        Config cfg = parse_args(argc, argv);
        if (cfg.trials <= 0 || cfg.k <= 0 || cfg.batch <= 0) {
            throw std::invalid_argument(
                "--trials, --k, and --batch must all be positive");
        }

        io::Dataset ds;
        int N = 0;
        int B = 0;

        if (rank == 0) {
            if (cfg.use_synthetic) {
                ds = make_synthetic_benchmark_dataset(cfg.syn_N, cfg.syn_B,
                                                      cfg.syn_d);
            } else {
                if (cfg.data_dir.empty()) {
                    throw std::invalid_argument(
                        "Error: --data <dir> required, or use --synthetic");
                }
                ds = io::load_dataset(cfg.data_dir, cfg.dataset);
            }

            N = (cfg.n_limit > 0) ? std::min(cfg.n_limit, ds.n_base) : ds.n_base;
            B = std::min(cfg.batch, ds.n_queries);
        }

        core::DistributedSearchConfig dist_cfg;
        dist_cfg.kernel = cfg.kernel;
        dist_cfg.root = 0;

        core::DistributedSearchIndex index(
            rank == 0 ? ds.base.data() : nullptr,
            N,
            rank == 0 ? ds.dim : 0,
            dist_cfg,
            MPI_COMM_WORLD);

        std::vector<double> totals;
        std::vector<double> bcasts;
        std::vector<double> locals;
        std::vector<double> gathers;
        std::vector<double> merges;
        core::DistributedSearchResult last;

        for (int t = 0; t < cfg.trials; ++t) {
            last = index.search(rank == 0 ? ds.queries.data() : nullptr, B, cfg.k);
            push_timing(totals, bcasts, locals, gathers, merges, last.timing);
        }

        if (rank == 0) {
            const Stats total = summarize(totals);
            const Stats bcast = summarize(bcasts);
            const Stats local = summarize(locals);
            const Stats gather = summarize(gathers);
            const Stats merge = summarize(merges);
            const double qps = B / (total.mean / 1000.0);

            float recall = -1.0f;
            if (!ds.gt.empty() && cfg.k <= ds.gt_k && N == ds.n_base) {
                recall = core::recall_at_k(last.result, ds.gt, ds.gt_k, B, cfg.k);
            }

            if (cfg.csv) {
                std::printf(
                    "%s,%s,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.4f\n",
                    ds.name.c_str(), cfg.kernel.c_str(), world_size,
                    N, index.dim(), cfg.k, B,
                    total.mean, total.min,
                    bcast.mean, local.mean, gather.mean, merge.mean,
                    last.timing.setup_ms, qps,
                    recall >= 0.0f ? recall : 0.0f);
            } else {
                std::printf("\n-- Distributed Benchmark ------------------------------\n");
                std::printf("  Dataset       : %s  (using %d / %d base vectors)\n",
                            ds.name.c_str(), N, ds.n_base);
                std::printf("  Kernel        : %s\n", cfg.kernel.c_str());
                std::printf("  Ranks         : %d\n", world_size);
                std::printf("  Local N rank0 : %d\n", last.local_N);
                std::printf("  Dim           : %d\n", index.dim());
                std::printf("  k             : %d\n", cfg.k);
                std::printf("  Batch         : %d queries\n", B);
                std::printf("  Trials        : %d\n", cfg.trials);
                std::printf("-------------------------------------------------------\n");
                std::printf("  setup         : %.2f ms\n", last.timing.setup_ms);
                std::printf("  scatter       : %.2f ms\n",
                            last.timing.scatter_ms);
                std::printf("  index build   : %.2f ms\n",
                            last.timing.index_build_ms);
                std::printf("  mean latency  : %.2f ms\n", total.mean);
                std::printf("  min latency   : %.2f ms\n", total.min);
                std::printf("  max latency   : %.2f ms\n", total.max);
                std::printf("  query bcast   : %.2f ms\n", bcast.mean);
                std::printf("  local search  : %.2f ms\n", local.mean);
                std::printf("  cand gather   : %.2f ms\n", gather.mean);
                std::printf("  root merge    : %.2f ms\n", merge.mean);
                std::printf("  throughput    : %.0f queries/s\n", qps);
                if (recall >= 0.0f) {
                    std::printf("  Recall@%-3d    : %.4f\n", cfg.k, recall);
                }
                std::printf("-------------------------------------------------------\n");
            }
        }

        MPI_Finalize();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return 1;
}
