#include "gpu_search.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(_err));        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace core {

__global__ void gpu_search_naive_kernel (const float* X, int N, int d, const float* Q, int B, float* S){
        int q = blockIdx.y;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        float acc = 0.0f;
        // compute the dot-product
        for (int j=0; j<d; j++){
            acc += X[i * d + j] * Q[q * d + j]; // flat indexing
        }
        S[q * N + i] = acc;
}


SearchResult gpu_search_naive(const float* X, int N, int d,
                              const float* Q, int B, int k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // STEP 1: allocate device buffers for the database X, query batch Q, and the output score matrix S, bc they are stored on RAM
    float* d_X = nullptr;
    float* d_Q = nullptr;
    float* d_S = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_X, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Q, B * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_S, B * N * sizeof(float)));

    // STEP 2: copy X and Q host to the device
    CUDA_CHECK(cudaMemcpy(d_X, X, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, Q, B * d * sizeof(float), cudaMemcpyHostToDevice));

    // STEP 3: launch kernel
    // we need one thread per dot product computation, there are B queries, and N rows for each database row.
    dim3 block(256);                         // blockDim.x covers N
    dim3 grid((N + 255) / 256, B);           // grid.x covers N in chunks of 256, grid.y covers B
    gpu_search_naive_kernel<<<grid, block>>>(d_X, N, d, d_Q, B, d_S);

    // STEP 4: cudaDeviceSynchronize + check launch error 
    CUDA_CHECK(cudaDeviceSynchronize());


    // STEP 5: copy of the B*N score matrix from the device to the host (To compute from the CPU)
    std::vector<float> scores_host((size_t)B * N);
    CUDA_CHECK(cudaMemcpy(scores_host.data(), d_S,
                          (size_t)B * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // STEP 6: CPU top-k over scores (reuse nth_element path)
    SearchResult result;
    result.indices.resize((size_t)B * k);
    result.scores.resize((size_t)B * k);

    std::vector<std::pair<float, int>> scored(N);
    for (int b = 0; b < B; ++b) {
        const float* row = scores_host.data() + (size_t)b * N;
        for (int i = 0; i < N; ++i) scored[i] = {row[i], i};

        // Partial sort: move top-k to the front, then sort those k descending.
        auto cmp = [](const std::pair<float,int>& a,
                      const std::pair<float,int>& b) { return a.first > b.first; };
        std::nth_element(scored.begin(), scored.begin() + k, scored.end(), cmp);
        std::sort(scored.begin(), scored.begin() + k, cmp);

        int*   out_idx = result.indices.data() + (size_t)b * k;
        float* out_scr = result.scores.data()  + (size_t)b * k;
        for (int j = 0; j < k; ++j) {
            out_idx[j] = scored[j].second;
            out_scr[j] = scored[j].first;
        }
    }

    // STEP 7: free device memory
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_S));

    // STEP 8: populate and return SearchResult
    auto t1 = std::chrono::high_resolution_clock::now();
    result.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

// ─── Tiled kernel ────────────────────────────────────────────────────────────
//
// Computes S = Q · Xᵀ  (B×N score matrix) using 32×32 thread blocks.
// Each block covers a TILE×TILE submatrix of S: rows b_base..b_base+TILE-1
// (queries) and columns n_base..n_base+TILE-1 (database vectors).
//
// Thread (ty, tx) owns output S[b_base+ty][n_base+tx].
//
// Shared memory layout (one tile step along d):
//   q_sm[ty][tx] = Q[b_base+ty][d_off+tx]   — row-major, warp reads
//                                               32 consecutive Q elements → coalesced
//   x_sm[ty][tx] = X[n_base+tx][d_off+ty]   — stored TRANSPOSED (dim × n).
//                                               Warp reads stride-d elements from X
//                                               (unavoidable with row-major X), but
//                                               the transposed layout means the compute
//                                               loop reads x_sm[k][tx=0..31], which are
//                                               32 consecutive elements → no bank conflicts.

static constexpr int TILE = 32;

__global__ static void gpu_search_tiled_kernel(
    const float* __restrict__ X,   // [N × d] database, row-major
    const float* __restrict__ Q,   // [B × d] queries,   row-major
    float*       __restrict__ S,   // [B × N] scores,    row-major
    int N, int d, int B)
{
    __shared__ float q_sm[TILE][TILE];  // q_sm[query_in_tile][d_in_tile]
    __shared__ float x_sm[TILE][TILE];  // x_sm[d_in_tile][n_in_tile]  ← transposed

    const int n_base = blockIdx.x * TILE;
    const int b_base = blockIdx.y * TILE;
    const int tx = threadIdx.x;   // n dimension (0..TILE)
    const int ty = threadIdx.y;   // b dimension (0..TILE)

    const int n = n_base + tx;
    const int b = b_base + ty;

    float acc = 0.0f;

    for (int d_off = 0; d_off < d; d_off += TILE) {
        // Load Q tile — coalesced: warp reads Q[b][d_off..d_off+31]
        q_sm[ty][tx] = (b < B && d_off + tx < d)
                       ? Q[(size_t)b * d + d_off + tx] : 0.0f;

        // Load X tile — non-coalesced (warp strides by d across db rows), but
        // stored transposed so the compute loop below has no bank conflicts.
        x_sm[ty][tx] = (n < N && d_off + ty < d)
                       ? X[(size_t)(n_base + tx) * d + d_off + ty] : 0.0f;

        __syncthreads();

        // Accumulate partial dot product for this d tile.
        // q_sm[ty][k]: all warp threads broadcast the same element — fine.
        // x_sm[k][tx]: warp reads x_sm[k][0..31], 32 consecutive floats — no bank conflicts.
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += q_sm[ty][k] * x_sm[k][tx];

        __syncthreads();
    }

    if (b < B && n < N)
        S[(size_t)b * N + n] = acc;
}

SearchResult gpu_search_tiled(const float* X, int N, int d,
                              const float* Q, int B, int k)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    float *d_X, *d_Q, *d_S;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q, (size_t)B * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, (size_t)B * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X, (size_t)N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, Q, (size_t)B * d * sizeof(float), cudaMemcpyHostToDevice));

    // One 32×32 block per TILE×TILE output submatrix.
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (B + TILE - 1) / TILE);
    gpu_search_tiled_kernel<<<grid, block>>>(d_X, d_Q, d_S, N, d, B);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> scores_host((size_t)B * N);
    CUDA_CHECK(cudaMemcpy(scores_host.data(), d_S,
                          (size_t)B * N * sizeof(float), cudaMemcpyDeviceToHost));

    SearchResult result;
    result.indices.resize((size_t)B * k);
    result.scores.resize((size_t)B * k);

    std::vector<std::pair<float, int>> scored(N);
    for (int b = 0; b < B; ++b) {
        const float* row = scores_host.data() + (size_t)b * N;
        for (int i = 0; i < N; ++i) scored[i] = {row[i], i};

        auto cmp = [](const std::pair<float,int>& a,
                      const std::pair<float,int>& b) { return a.first > b.first; };
        std::nth_element(scored.begin(), scored.begin() + k, scored.end(), cmp);
        std::sort(scored.begin(), scored.begin() + k, cmp);

        int*   out_idx = result.indices.data() + (size_t)b * k;
        float* out_scr = result.scores.data()  + (size_t)b * k;
        for (int j = 0; j < k; ++j) {
            out_idx[j] = scored[j].second;
            out_scr[j] = scored[j].first;
        }
    }

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_S));

    auto t1 = std::chrono::high_resolution_clock::now();
    result.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

} // namespace core
