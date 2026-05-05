#pragma once

#include "../core/cpu_search.hpp"

namespace core {

// maximum inner-product search (MIPS) on a single GPU.
//
// X : database, row-major float[N * d]   (host memory)
// N : number of database vectors
// d : embedding dimension
// Q : query batch, row-major float[B * d] (host memory)
// B : number of queries
// k : number of nearest neighbours to return
//
// Returns results matching cpu_search: indices[b*k + j] is the j-th nearest
// neighbour of query b, scores[b*k + j] is the corresponding inner product,
// sorted in descending order.
//
// one CUDA thread per (query, row) dot product, CPU nth_element for top-k.
SearchResult gpu_search_naive(const float* X, int N, int d,
                              const float* Q, int B, int k);

// Tiled GPU search.
//
// This computes the same B-by-N score matrix as the naive version, but each
// CUDA block works on a 32-by-32 square of scores at a time. The block loads a
// small piece of Q and a small piece of X into shared memory, reuses those
// values to compute dot products, then moves to the next 32 dimensions.
//
// Top-k is still computed on the CPU with nth_element, matching the naive path.
SearchResult gpu_search_tiled(const float* X, int N, int d,
                              const float* Q, int B, int k);

// INT8 GPU search.
//
// This keeps the same one-thread-per-dot-product structure as the naive CUDA
// kernel, but the database X is quantized on the CPU before being copied to the
// GPU. Each row of X gets one scale factor, and the kernel reads int8 values
// instead of float values for X.
//
// Q stays float, scores stay float, and top-k is still computed on the CPU.
SearchResult gpu_search_int8(const float* X, int N, int d,
                             const float* Q, int B, int k);

}
