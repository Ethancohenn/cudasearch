//  main_bench — configurable benchmark driver for CUDAsearch
//
//  Usage:
//    ./bench [options]
//
//  Options:
//    --data <dir>         Path to dataset directory
//    --dataset <name>     Dataset name (default: sift1m)
//    --kernel  <name>     Kernel: cpu  [future: naive, tiled, int8]
//    --k <int>            Number of neighbours (default: 10)
//    --batch <int>        Query batch size (default: 100)
//    --n <int>            Max database size (-1 = full, default: -1)
//    --trials <int>       Number of timed runs (default: 5)
//    --synthetic          Use synthetic random data instead of disk
//    --syn-N <int>        Synthetic base size (default: 100000)
//    --syn-B <int>        Synthetic query count (default: 1000)
//    --syn-d <int>        Synthetic dimension (default: 128)
//    --csv                Print result as CSV row (for scripting)
//
//  Example:
//    ./bench --synthetic --syn-N 1000000 --syn-d 128 --k 10 --batch 64
// ─────────────────────────────────────────────────────────────────────────────

#include "core/cpu_search.hpp"
#include "io/fvecs_loader.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Config {
    std::string data_dir;
    std::string dataset   = "sift1m";
    std::string kernel    = "cpu";
    int k                 = 10;
    int batch             = 100;
    int n_limit           = -1;      // -1 = full
    int trials            = 5;
    bool use_synthetic    = false;
    int syn_N             = 100'000;
    int syn_B             = 1000;
    int syn_d             = 128;
    bool csv              = false;
};

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* s) { return std::strcmp(argv[i], s) == 0; };
        auto next_int = [&]() { return std::atoi(argv[++i]); };
        auto next_str = [&]() -> std::string { return argv[++i]; };

        if      (is("--data"))      cfg.data_dir       = next_str();
        else if (is("--dataset"))   cfg.dataset        = next_str();
        else if (is("--kernel"))    cfg.kernel         = next_str();
        else if (is("--k"))         cfg.k              = next_int();
        else if (is("--batch"))     cfg.batch          = next_int();
        else if (is("--n"))         cfg.n_limit        = next_int();
        else if (is("--trials"))    cfg.trials         = next_int();
        else if (is("--synthetic")) cfg.use_synthetic  = true;
        else if (is("--syn-N"))     cfg.syn_N          = next_int();
        else if (is("--syn-B"))     cfg.syn_B          = next_int();
        else if (is("--syn-d"))     cfg.syn_d          = next_int();
        else if (is("--csv"))       cfg.csv            = true;
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            std::exit(1);
        }
    }
    return cfg;
}

// ─── Timing helpers ───────────────────────────────────────────────────────────

struct Stats {
    double mean_ms;
    double min_ms;
    double max_ms;
    double qps;      // queries per second (mean)
    float  recall;
};

static Stats run_trials(const Config& cfg,
                        const float* X, int N, int d,
                        const float* Q, int B, int k,
                        const int* gt, int gt_k) {
    std::vector<double> times;
    core::SearchResult last_result;

    for (int t = 0; t < cfg.trials; ++t) {
        if (cfg.kernel == "cpu") {
            last_result = core::cpu_search(X, N, d, Q, B, k);
        } else {
            fprintf(stderr, "Unknown kernel: %s (only 'cpu' implemented for now)\n",
                    cfg.kernel.c_str());
            std::exit(1);
        }
        times.push_back(last_result.wall_ms);
    }

    Stats s;
    s.min_ms  = *std::min_element(times.begin(), times.end());
    s.max_ms  = *std::max_element(times.begin(), times.end());
    s.mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    s.qps     = B / (s.mean_ms / 1000.0);
    s.recall  = gt ? core::recall_at_k(last_result, std::vector<int>(gt, gt + (size_t)B * gt_k), gt_k, B, k)
                   : -1.f;
    return s;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    io::Dataset ds;
    if (cfg.use_synthetic) {
        ds = io::make_random_dataset(cfg.syn_N, cfg.syn_B, cfg.syn_d, cfg.k);
    } else {
        if (cfg.data_dir.empty()) {
            fprintf(stderr, "Error: --data <dir> required (or use --synthetic)\n");
            return 1;
        }
        ds = io::load_dataset(cfg.data_dir, cfg.dataset);
    }

    // Optionally cap database size for sweep experiments
    int N = (cfg.n_limit > 0) ? std::min(cfg.n_limit, ds.n_base) : ds.n_base;
    int B = std::min(cfg.batch, ds.n_queries);

    if (!cfg.csv) {
        printf("\n── Benchmark ────────────────────────────────────────────────\n");
        printf("  Dataset  : %s  (using %d / %d base vectors)\n",
               ds.name.c_str(), N, ds.n_base);
        printf("  Kernel   : %s\n", cfg.kernel.c_str());
        printf("  Dim      : %d\n", ds.dim);
        printf("  k        : %d\n", cfg.k);
        printf("  Batch    : %d queries\n", B);
        printf("  Trials   : %d\n", cfg.trials);
        printf("─────────────────────────────────────────────────────────────\n");
    }

    const int* gt = ds.gt.empty() ? nullptr : ds.gt.data();
    int gt_k      = ds.gt_k;

    Stats s = run_trials(cfg,
                         ds.base.data(), N, ds.dim,
                         ds.queries.data(), B, cfg.k,
                         gt, gt_k);

    if (cfg.csv) {
        // CSV row: dataset,kernel,N,d,k,B,mean_ms,min_ms,qps,recall
        printf("%s,%s,%d,%d,%d,%d,%.2f,%.2f,%.1f,%.4f\n",
               ds.name.c_str(), cfg.kernel.c_str(),
               N, ds.dim, cfg.k, B,
               s.mean_ms, s.min_ms, s.qps,
               s.recall >= 0 ? s.recall : 0.f);
    } else {
        printf("\n  mean latency : %.2f ms\n", s.mean_ms);
        printf("  min  latency : %.2f ms\n", s.min_ms);
        printf("  max  latency : %.2f ms\n", s.max_ms);
        printf("  throughput   : %.0f queries/s\n", s.qps);
        if (s.recall >= 0)
            printf("  Recall@%-3d   : %.4f\n", cfg.k, s.recall);
        printf("─────────────────────────────────────────────────────────────\n");
    }

    return 0;
}
