//  test_recall — validate cpu_search correctness
//
//  Usage:
//    ./test_recall                          # synthetic only (fast)
//    ./test_recall --data <dir> sift1m      # also test on SIFT1M
//    ./test_recall --data <dir> gist1m      # also test on GIST1M
//
//  The binary exits 0 on pass, 1 on failure.
// ─────────────────────────────────────────────────────────────────────────────

#include "core/cpu_search.hpp"
#include "io/fvecs_loader.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ─── Helpers ──────────────────────────────────────────────────────────────────

static bool check(bool cond, const char* msg) {
    if (!cond) printf("[FAIL] %s\n", msg);
    return cond;
}

// ─── Test 1: self-consistency on synthetic data ───────────────────────────────
// cpu_search against its own ground truth must give Recall@k = 1.0

static bool test_synthetic_recall() {
    printf("\n=== Test: synthetic recall (N=5000, d=128, B=100, k=10) ===\n");

    int N = 5000, d = 128, B = 100, k = 10;
    auto ds = io::make_random_dataset(N, B, d, k, /*seed=*/42);

    auto result = core::cpu_search(ds.base, N, d, ds.queries, B, k);

    float r = core::recall_at_k(result, ds.gt, k, B, k);
    printf("  Recall@%d = %.4f  (expected 1.0000)\n", k, r);

    core::print_result_summary("cpu_search synthetic", result, ds.gt, k, B, k);
    return check(r >= 0.9999f, "Recall@10 on synthetic data < 1.0");
}

// ─── Test 2: trivial 3-vector sanity check ───────────────────────────────────

static bool test_trivial() {
    printf("\n=== Test: trivial 3-vector sanity check ===\n");

    // d=4, N=3 vectors
    float X[] = {1,0,0,0,   // index 0 — e1
                 0,1,0,0,   // index 1 — e2
                 0,0,1,0};  // index 2 — e3

    // Query: close to index 1 (e2)
    float Q[] = {0.1f, 0.9f, 0.0f, 0.0f};

    auto r = core::cpu_search(X, 3, 4, Q, 1, /*k=*/3);

    printf("  Top-3 indices: %d %d %d\n",
           r.indices[0], r.indices[1], r.indices[2]);
    printf("  Top-3 scores : %.3f %.3f %.3f\n",
           r.scores[0], r.scores[1], r.scores[2]);

    bool ok = (r.indices[0] == 1);  // best match must be e2
    return check(ok, "Trivial test: top-1 is not index 1");
}

// ─── Test 3: recall against real dataset ground truth ─────────────────────────

static bool test_real_dataset(const std::string& data_dir,
                              const std::string& name,
                              int max_queries = 1000) {
    printf("\n=== Test: real dataset '%s' (up to %d queries) ===\n",
           name.c_str(), max_queries);

    io::Dataset ds;
    try {
        ds = io::load_dataset(data_dir, name);
    } catch (const std::exception& e) {
        printf("  [SKIP] Could not load '%s': %s\n", name.c_str(), e.what());
        return true;  // skip, not fail
    }

    int B = std::min(max_queries, ds.n_queries);

    for (int k : {1, 10, 100}) {
        if (k > ds.gt_k) continue;

        auto result = core::cpu_search(
            ds.base.data(),    ds.n_base,    ds.dim,
            ds.queries.data(), B,            k);

        core::print_result_summary(name + " cpu k=" + std::to_string(k),
                                   result, ds.gt, ds.gt_k, B, k);

        float r = core::recall_at_k(result, ds.gt, ds.gt_k, B, k);
        // Exhaustive MIPS should always hit 1.0 against its own benchmark GT
        if (!check(r >= 0.999f,
                   ("Recall@" + std::to_string(k) + " < 0.999 on " + name).c_str()))
            return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string data_dir;
    std::vector<std::string> datasets;

    // Parse: --data <dir>  [dataset_name ...]
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else {
            datasets.push_back(argv[i]);
        }
    }
    if (!data_dir.empty() && datasets.empty()) {
        // Default: try both SIFT1M and GIST1M
        datasets = {"sift1m", "gist1m"};
    }

    int n_fail = 0;
    if (!test_trivial())           ++n_fail;
    if (!test_synthetic_recall())  ++n_fail;
    for (auto& ds : datasets)
        if (!test_real_dataset(data_dir, ds)) ++n_fail;

    printf("\n%s — %d test(s) failed.\n",
           n_fail == 0 ? "PASSED" : "FAILED", n_fail);
    return n_fail > 0 ? 1 : 0;
}
