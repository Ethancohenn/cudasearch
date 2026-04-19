// Implementation of the ANN dataset loading layer.
//
// This file reads the TexMex-style binary vector formats used by datasets like
// SIFT1M and GIST1M, validates their repeated dimension headers, and converts
// them into flat row-major C++ arrays for the search and benchmark code.
// It also provides a synthetic random dataset generator for tests.

#include "io/fvecs_loader.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace io {

// ─── Internal helpers ─────────────────────────────────────────────────────────

static FILE* open_or_die(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("Cannot open file: " + path);
    return f;
}

// Count the number of vectors in an *vecs file (any element type of size
// elem_bytes) by reading the first dimension header and seeking to the end.
static int count_vectors(FILE* f, int elem_bytes) {
    int32_t dim;
    if (std::fread(&dim, sizeof(int32_t), 1, f) != 1)
        throw std::runtime_error("File appears empty");
    std::fseek(f, 0, SEEK_END);
    long file_size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    long record_size = sizeof(int32_t) + (long)dim * elem_bytes;
    if (file_size % record_size != 0)
        throw std::runtime_error(
            "File size is not a multiple of the expected record size. "
            "Corrupted file?");
    return (int)(file_size / record_size);
}

// ─── Public loaders ───────────────────────────────────────────────────────────

std::vector<float> load_fvecs(const std::string& path,
                               int& n_vectors, int& dim) {
    FILE* f = open_or_die(path);

    // Read dimension from first record header
    int32_t d;
    if (std::fread(&d, sizeof(int32_t), 1, f) != 1)
        throw std::runtime_error("load_fvecs: empty file " + path);
    std::fseek(f, 0, SEEK_SET);

    int n = count_vectors(f, sizeof(float));
    n_vectors = n;
    dim = (int)d;

    std::vector<float> data((size_t)n * d);

    for (int i = 0; i < n; ++i) {
        int32_t d_check;
        if (std::fread(&d_check, sizeof(int32_t), 1, f) != 1 || d_check != d)
            throw std::runtime_error(
                "load_fvecs: inconsistent dimension at vector " +
                std::to_string(i));
        if (std::fread(data.data() + (size_t)i * d, sizeof(float), d, f) !=
            (size_t)d)
            throw std::runtime_error(
                "load_fvecs: short read at vector " + std::to_string(i));
    }
    std::fclose(f);
    return data;
}

std::vector<int> load_ivecs(const std::string& path,
                             int& n_vectors, int& dim) {
    FILE* f = open_or_die(path);

    int32_t d;
    if (std::fread(&d, sizeof(int32_t), 1, f) != 1)
        throw std::runtime_error("load_ivecs: empty file " + path);
    std::fseek(f, 0, SEEK_SET);

    int n = count_vectors(f, sizeof(int32_t));
    n_vectors = n;
    dim = (int)d;

    std::vector<int> data((size_t)n * d);

    for (int i = 0; i < n; ++i) {
        int32_t d_check;
        if (std::fread(&d_check, sizeof(int32_t), 1, f) != 1 || d_check != d)
            throw std::runtime_error(
                "load_ivecs: inconsistent dimension at vector " +
                std::to_string(i));
        if (std::fread(data.data() + (size_t)i * d, sizeof(int32_t), d, f) !=
            (size_t)d)
            throw std::runtime_error(
                "load_ivecs: short read at vector " + std::to_string(i));
    }
    std::fclose(f);
    return data;
}

std::vector<float> load_bvecs(const std::string& path,
                               int& n_vectors, int& dim) {
    FILE* f = open_or_die(path);

    int32_t d;
    if (std::fread(&d, sizeof(int32_t), 1, f) != 1)
        throw std::runtime_error("load_bvecs: empty file " + path);
    std::fseek(f, 0, SEEK_SET);

    int n = count_vectors(f, sizeof(uint8_t));
    n_vectors = n;
    dim = (int)d;

    std::vector<uint8_t> raw((size_t)n * d);
    for (int i = 0; i < n; ++i) {
        int32_t d_check;
        if (std::fread(&d_check, sizeof(int32_t), 1, f) != 1 || d_check != d)
            throw std::runtime_error(
                "load_bvecs: inconsistent dimension at vector " +
                std::to_string(i));
        if (std::fread(raw.data() + (size_t)i * d, sizeof(uint8_t), d, f) !=
            (size_t)d)
            throw std::runtime_error(
                "load_bvecs: short read at vector " + std::to_string(i));
    }
    std::fclose(f);

    // Cast to float
    std::vector<float> data(raw.size());
    for (size_t j = 0; j < raw.size(); ++j)
        data[j] = static_cast<float>(raw[j]);
    return data;
}

// ─── Dataset loader ───────────────────────────────────────────────────────────

Dataset load_dataset(const std::string& dir, const std::string& name) {
    Dataset ds;
    ds.name = name;

    std::string base_path = dir + "/" + name + "_base.fvecs";
    std::string query_path = dir + "/" + name + "_query.fvecs";
    std::string gt_path    = dir + "/" + name + "_groundtruth.ivecs";

    ds.base    = load_fvecs(base_path,  ds.n_base,    ds.dim);
    ds.queries = load_fvecs(query_path, ds.n_queries, ds.dim);
    ds.gt      = load_ivecs(gt_path,    ds.n_queries, ds.gt_k);

    printf("[io] Loaded %s: %d base vectors, %d queries, dim=%d, gt_k=%d\n",
           name.c_str(), ds.n_base, ds.n_queries, ds.dim, ds.gt_k);
    return ds;
}

// ─── Synthetic dataset ────────────────────────────────────────────────────────

// Brute-force top-k inner product for ground-truth generation (tiny datasets only)
static void tiny_search(const float* X, int N, int d,
                         const float* q, int k, std::vector<int>& idx) {
    std::vector<std::pair<float, int>> scores(N);
    for (int i = 0; i < N; ++i) {
        float dot = 0.f;
        for (int j = 0; j < d; ++j) dot += q[j] * X[(size_t)i * d + j];
        scores[i] = {dot, i};
    }
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    idx.resize(k);
    for (int j = 0; j < k; ++j) idx[j] = scores[j].second;
}

Dataset make_random_dataset(int n_base, int n_queries, int dim, int gt_k,
                            unsigned seed) {
    Dataset ds;
    ds.name      = "random";
    ds.n_base    = n_base;
    ds.n_queries = n_queries;
    ds.dim       = dim;
    ds.gt_k      = gt_k;

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);

    auto make_matrix = [&](int rows) {
        std::vector<float> M((size_t)rows * dim);
        for (auto& v : M) v = nd(rng);
        // L2-normalise each row so that IP = cosine similarity
        for (int i = 0; i < rows; ++i) {
            float* row = M.data() + (size_t)i * dim;
            float norm = 0.f;
            for (int j = 0; j < dim; ++j) norm += row[j] * row[j];
            norm = std::sqrt(norm);
            for (int j = 0; j < dim; ++j) row[j] /= norm;
        }
        return M;
    };

    ds.base    = make_matrix(n_base);
    ds.queries = make_matrix(n_queries);

    // Ground truth via brute force (fine for small datasets)
    ds.gt.resize((size_t)n_queries * gt_k);
    for (int q = 0; q < n_queries; ++q) {
        std::vector<int> idx;
        tiny_search(ds.base.data(), n_base, dim,
                    ds.queries.data() + (size_t)q * dim, gt_k, idx);
        for (int j = 0; j < gt_k; ++j)
            ds.gt[(size_t)q * gt_k + j] = idx[j];
    }

    printf("[io] Generated random dataset: %d base, %d queries, dim=%d, gt_k=%d\n",
           n_base, n_queries, dim, gt_k);
    return ds;
}

} // namespace io
