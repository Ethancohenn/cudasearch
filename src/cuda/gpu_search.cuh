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

} 
