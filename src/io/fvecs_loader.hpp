#pragma once
#include <string>
#include <vector>
#include <cstdint>

// Dataset loading utilities for Approximate Nearest Neighbor (ANN) benchmark files.
//
// CUDAsearch works on dense vector datasets: a large database of vectors,
// a smaller set of query vectors, and optional ground-truth nearest-neighbour
// indices used to measure recall. Benchmark datasets such as SIFT1M and GIST1M
// store these arrays on disk in compact binary formats instead of text files.
//
// This header defines the IO layer that translates those binary files into
// normal C++ containers. The search code does not need to know how .fvecs,
// .ivecs, or .bvecs files are encoded; it only receives flat row-major arrays:
//
//   data[i * dim + j] == component j of vector i
//
// File types:
//   .fvecs: float vectors, used for base/query embeddings.
//   .ivecs: integer vectors, used here for ground-truth neighbour indices.
//   .bvecs: byte vectors, converted to float when loaded.
//
// The Dataset struct bundles base vectors, query vectors, dimensions, and
// ground truth so benchmarks/tests can pass around one object.

namespace io {

// ─── fvecs / ivecs / bvecs format ─────────────────────────────────────────────
//  Each vector:  [int32_t dim] [dim × T values]
//  The dimension field is repeated for every vector.
//  See: http://corpus-texmex.irisa.fr/
// ──────────────────────────────────────────────────────────────────────────────

// Load an .fvecs file.
// Returns a flat row-major matrix of shape (n_vectors, dim).
// n_vectors and dim are set on output.
std::vector<float> load_fvecs(const std::string& path,
                               int& n_vectors, int& dim);

// Load an .ivecs file (e.g., ground-truth nearest-neighbour indices).
std::vector<int> load_ivecs(const std::string& path,
                             int& n_vectors, int& dim);

// Load a .bvecs file (byte vectors, e.g., SIFT descriptors before conversion).
// Values are cast to float on load.
std::vector<float> load_bvecs(const std::string& path,
                               int& n_vectors, int& dim);

// ─── Dataset bundle ───────────────────────────────────────────────────────────
struct Dataset {
    // Database (corpus) vectors — shape (n_base, dim)
    std::vector<float> base;
    int n_base = 0;

    // Query vectors — shape (n_queries, dim)
    std::vector<float> queries;
    int n_queries = 0;

    // Dimension (same for base and queries)
    int dim = 0;

    // Ground-truth top-k indices — shape (n_queries, gt_k)
    // gt[q * gt_k + j] = index of the j-th nearest neighbour of query q
    std::vector<int> gt;
    int gt_k = 0;

    // Human-readable name ("SIFT1M", "GIST1M", ...)
    std::string name;
};

// Convenience loader for a standard ANN benchmark dataset directory.
// Expects:
//   <dir>/<name>_base.fvecs
//   <dir>/<name>_query.fvecs
//   <dir>/<name>_groundtruth.ivecs
// Base and query vectors are L2-normalized after loading, so inner product
// search is equivalent to cosine similarity on the loaded dataset.
Dataset load_dataset(const std::string& dir, const std::string& name);

// Generate a synthetic random dataset (unit-normalised, so IP = cosine).
// Useful for quick unit tests without downloading real data.
Dataset make_random_dataset(int n_base, int n_queries, int dim, int gt_k,
                            unsigned seed = 42);

} // namespace io
