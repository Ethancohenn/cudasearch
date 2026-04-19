#include "core/cpu_search.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace core {

// ─── Inner product ────────────────────────────────────────────────────────────
// Manually unrolled so the compiler auto-vectorises reliably even without -O3.

static float dot_product(const float* __restrict__ a,
                          const float* __restrict__ b, int d) {
    float acc = 0.f;
    // 8-wide unroll — compiler maps to AVX2 when -march=native
    int i = 0;
    for (; i + 7 < d; i += 8) {
        acc += a[i+0]*b[i+0] + a[i+1]*b[i+1] +
               a[i+2]*b[i+2] + a[i+3]*b[i+3] +
               a[i+4]*b[i+4] + a[i+5]*b[i+5] +
               a[i+6]*b[i+6] + a[i+7]*b[i+7];
    }
    for (; i < d; ++i) acc += a[i] * b[i];
    return acc;
}

// ─── Core search ──────────────────────────────────────────────────────────────

SearchResult cpu_search(const float* X, int N, int d,
                        const float* Q, int B, int k) {
    if (k > N)
        throw std::invalid_argument("k (" + std::to_string(k) +
                                    ") > N (" + std::to_string(N) + ")");

    SearchResult res;
    res.indices.resize((size_t)B * k);
    res.scores.resize((size_t)B * k);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Each query is independent → parallelise over B.
#ifdef _OPENMP
#pragma omp parallel
    {
        // Thread-local score buffer (avoids per-thread allocation in the loop)
        std::vector<std::pair<float, int>> scores(N);

#pragma omp for schedule(dynamic, 1)
        for (int b = 0; b < B; ++b) {
#else
        {
            std::vector<std::pair<float, int>> scores(N);
            for (int b = 0; b < B; ++b) {
#endif
            const float* q = Q + (size_t)b * d;

            // ── Score every database vector ──────────────────────────────────
            for (int i = 0; i < N; ++i)
                scores[i] = {dot_product(q, X + (size_t)i * d, d), i};

            // ── Partial sort: move top-k to front in O(N + k log k) ──────────
            // nth_element brings the k largest to scores[0..k-1] (unsorted).
            std::nth_element(scores.begin(),
                             scores.begin() + k,
                             scores.end(),
                             [](const auto& a, const auto& b) {
                                 return a.first > b.first;
                             });
            // Sort those k elements so that index 0 is the best.
            std::sort(scores.begin(), scores.begin() + k,
                      [](const auto& a, const auto& b) {
                          return a.first > b.first;
                      });

            // ── Write result ─────────────────────────────────────────────────
            int* out_idx   = res.indices.data() + (size_t)b * k;
            float* out_scr = res.scores.data()  + (size_t)b * k;
            for (int j = 0; j < k; ++j) {
                out_idx[j] = scores[j].second;
                out_scr[j] = scores[j].first;
            }
#ifdef _OPENMP
        }
    }
#else
            }
        }
#endif

    auto t1 = std::chrono::high_resolution_clock::now();
    res.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

SearchResult cpu_search(const std::vector<float>& X, int N, int d,
                        const std::vector<float>& Q, int B, int k) {
    return cpu_search(X.data(), N, d, Q.data(), B, k);
}

// ─── Recall ───────────────────────────────────────────────────────────────────

float recall_at_k(const int* pred_idx, int k_pred,
                  const int* true_idx, int k_true,
                  int B, int k) {
    assert(k <= k_pred && "k must be <= k_pred");
    assert(k <= k_true && "k must be <= k_true");

    double total = 0.0;

    for (int b = 0; b < B; ++b) {
        const int* pred = pred_idx + (size_t)b * k_pred;
        const int* gt   = true_idx  + (size_t)b * k_true;

        // Build a hash set of the true top-k
        std::unordered_set<int> gt_set(gt, gt + k);

        int hits = 0;
        for (int j = 0; j < k; ++j)
            if (gt_set.count(pred[j])) ++hits;

        total += (double)hits / k;
    }

    return (float)(total / B);
}

float recall_at_k(const SearchResult& result,
                  const std::vector<int>& gt, int gt_k,
                  int B, int k) {
    int k_pred = (int)(result.indices.size() / B);
    return recall_at_k(result.indices.data(), k_pred,
                       gt.data(), gt_k,
                       B, k);
}

// ─── Pretty-print ─────────────────────────────────────────────────────────────

void print_result_summary(const std::string& label,
                          const SearchResult& result,
                          const std::vector<int>& gt, int gt_k,
                          int B, int k) {
    float r = recall_at_k(result, gt, gt_k, B, k);
    double qps = B / (result.wall_ms / 1000.0);
    printf("[%-20s]  Recall@%-3d = %.4f  |  %.1f ms total  |  %.0f queries/s\n",
           label.c_str(), k, r, result.wall_ms, qps);
}

} // namespace core
