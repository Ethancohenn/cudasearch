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

// 32×32 tiled kernel: each block computes a TILE×TILE submatrix of S = Q·Xᵀ.
// Q tile loaded coalesced; X tile non-coalesced (unavoidable, row-major X) but
// stored transposed in shared memory to eliminate bank conflicts in the compute
// loop. CPU nth_element for top-k.
SearchResult gpu_search_tiled(const float* X, int N, int d,
                              const float* Q, int B, int k);

}
