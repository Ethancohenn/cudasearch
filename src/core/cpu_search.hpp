#pragma once
#include <vector>
#include <string>

namespace core {

// ─── Search result ────────────────────────────────────────────────────────────

struct SearchResult {
    std::vector<int>   indices;  // shape (n_queries * k)
    std::vector<float> scores;  // shape (n_queries * k), descending order
    double wall_ms = 0.0;       // total wall time in milliseconds
};

// ─── CPU baseline ─────────────────────────────────────────────────────────────
//
// Exhaustive maximum inner-product search (MIPS).
//
// X       : database, row-major float[N * d]
// N       : number of database vectors
// d       : embedding dimension
// Q       : query batch, row-major float[B * d]
// B       : number of queries in the batch
// k       : number of nearest neighbours to return
//
// Returns results where indices[b*k + j] is the j-th nearest neighbour of
// query b, and scores[b*k + j] is the corresponding inner product.
// Results are sorted in descending score order (best match first).
//
// Thread-parallelised over B queries with OpenMP (if compiled with -fopenmp).
// ─────────────────────────────────────────────────────────────────────────────
SearchResult cpu_search(const float* X, int N, int d,
                        const float* Q, int B, int k);

// Convenience overload that takes the flat vectors directly.
SearchResult cpu_search(const std::vector<float>& X, int N, int d,
                        const std::vector<float>& Q, int B, int k);

// ─── Recall evaluation ────────────────────────────────────────────────────────
//
// Computes Recall@k: fraction of the true top-k that appear in the predicted
// top-k, averaged over all B queries.
//
// pred_idx : predicted indices, shape (B * k_pred), k_pred >= k
// true_idx : ground-truth indices, shape (B * k_true), k_true >= k
// B        : number of queries
// k        : cutoff to evaluate
//
// Returns a value in [0, 1]. A correct exhaustive MIPS over FP32 data should
// return exactly 1.0 against itself (modulo ties).
// ─────────────────────────────────────────────────────────────────────────────
float recall_at_k(const int* pred_idx, int k_pred,
                  const int* true_idx, int k_true,
                  int B, int k);

// Convenience wrapper
float recall_at_k(const SearchResult& result,
                  const std::vector<int>& gt, int gt_k,
                  int B, int k);

// ─── Pretty-print helpers ──────────────────────────────────────────────────────
void print_result_summary(const std::string& label,
                          const SearchResult& result,
                          const std::vector<int>& gt, int gt_k,
                          int B, int k);

} // namespace core
