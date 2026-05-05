#include "gpu_search.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
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

// Tiled kernel
//
// The naive kernel uses one thread for one complete dot product. This tiled
// kernel keeps that same idea: one thread still owns one score S[b][n], where
// b is a query and n is a database vector.
//
// The difference is that a 32-by-32 block of threads works together. During
// each pass through the embedding dimension d, the block loads:
//   - 32 query values into q_sm
//   - 32 database-vector values into x_sm
//
// After the values are in shared memory, every thread adds 32 multiply-adds to
// its own dot product. Then the block advances to the next 32 dimensions.
//
// x_sm is stored transposed: x_sm[dimension_inside_tile][database_row_inside_tile].
// That makes the compute loop easy to read: q_sm[ty][j] and x_sm[j][tx] are the
// two numbers for the same dimension j.

static constexpr int TILE = 32;

__global__ void gpu_search_tiled_kernel(const float* X, int N, int d,
                                        const float* Q, int B, float* S)
{
    // q_sm holds query values for this block.
    // x_sm holds database values for this block, stored transposed.
    __shared__ float q_sm[TILE][TILE];
    __shared__ float x_sm[TILE][TILE];

    int n_base = blockIdx.x * TILE;  // first database row handled by this block
    int b_base = blockIdx.y * TILE;  // first query handled by this block
    int tx = threadIdx.x;            // column inside the tile, maps to database row
    int ty = threadIdx.y;            // row inside the tile, maps to query

    int n = n_base + tx;
    int b = b_base + ty;

    float acc = 0.0f;

    for (int d_off = 0; d_off < d; d_off += TILE) {
        // Load one 32-wide slice of the query vector for this thread's query.
        if ((b < B) && (d_off + tx < d)) {
            q_sm[ty][tx] = Q[(size_t)b * d + (d_off + tx)];
        } else {
            q_sm[ty][tx] = 0.0f;
        }

        // Load one 32-wide slice of the database vector for this thread's row.
        // The indices are flipped in shared memory: ty selects the
        // dimension, tx selects the database row.
        if ((n < N) && (d_off + ty < d)) {
            x_sm[ty][tx] = X[(size_t)n * d + (d_off + ty)];
        } else {
            x_sm[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Add this 32-dimensional slice to the dot product owned by this thread.
        #pragma unroll
        for (int j = 0; j < TILE; ++j) {
            acc += q_sm[ty][j] * x_sm[j][tx];
        }

        // Wait before reusing the shared-memory arrays for the next d slice.
        __syncthreads();
    }

    if (b < B && n < N) {
        S[(size_t)b * N + n] = acc;
    }
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

    // One 32-by-32 block computes one 32-by-32 tile of the score matrix.
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (B + TILE - 1) / TILE);
    gpu_search_tiled_kernel<<<grid, block>>>(d_X, N, d, d_Q, B, d_S);
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

// INT8 kernel
//
// This is deliberately close to the naive kernel:
//   - one thread owns one score S[q][i]
//   - the thread loops over all d dimensions
//   - top-k is still done later on the CPU
//
// The only difference is the database format. Xq stores one signed int8 value
// per database coordinate, and X_scale[i] tells us how to convert row i back to
// approximate floats:
//
//     X[i][j] ~= X_scale[i] * Xq[i][j]
//
// Q remains float, so the kernel accumulates int8 * float products and applies
// the row scale once at the end.

__global__ void gpu_search_int8_kernel(const int8_t* Xq, const float* X_scale,
                                       int N, int d,
                                       const float* Q, int B, float* S)
{
    int q = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float acc = 0.0f;
    for (int j = 0; j < d; ++j) {
        acc += (float)Xq[(size_t)i * d + j] * Q[(size_t)q * d + j];
    }

    S[(size_t)q * N + i] = acc * X_scale[i];
}

static void quantize_X_int8(const float* X, int N, int d,
                            std::vector<int8_t>& Xq,
                            std::vector<float>& X_scale)
{
    Xq.resize((size_t)N * d);
    X_scale.resize(N);

    for (int i = 0; i < N; ++i) {
        const float* row = X + (size_t)i * d;

        // Use one scale per database vector. This keeps the quantization easy to
        // understand and gives each row its own dynamic range.
        float max_abs = 0.0f;
        for (int j = 0; j < d; ++j) {
            max_abs = std::max(max_abs, std::fabs(row[j]));
        }

        float scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
        X_scale[i] = scale;

        for (int j = 0; j < d; ++j) {
            int q = (int)std::lround(row[j] / scale);
            q = std::max(-127, std::min(127, q));
            Xq[(size_t)i * d + j] = (int8_t)q;
        }
    }
}

SearchResult gpu_search_int8(const float* X, int N, int d,
                             const float* Q, int B, int k)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    // STEP 1: quantize the database on the CPU.
    // Xq is 4x smaller than X because each coordinate is one byte instead of
    // one float. X_scale stores one float scale per database vector.
    std::vector<int8_t> Xq_host;
    std::vector<float> X_scale_host;
    quantize_X_int8(X, N, d, Xq_host, X_scale_host);

    // STEP 2: allocate device buffers.
    int8_t* d_Xq = nullptr;
    float* d_X_scale = nullptr;
    float* d_Q = nullptr;
    float* d_S = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_Xq, (size_t)N * d * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_X_scale, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Q, (size_t)B * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_S, (size_t)B * N * sizeof(float)));

    // STEP 3: copy quantized X, row scales, and Q to the device.
    CUDA_CHECK(cudaMemcpy(d_Xq, Xq_host.data(),
                          (size_t)N * d * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X_scale, X_scale_host.data(),
                          (size_t)N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, Q, (size_t)B * d * sizeof(float),
                          cudaMemcpyHostToDevice));

    // STEP 4: launch the same grid shape as the naive kernel.
    // Each thread computes one approximate dot product.
    dim3 block(256);
    dim3 grid((N + 255) / 256, B);
    gpu_search_int8_kernel<<<grid, block>>>(d_Xq, d_X_scale, N, d, d_Q, B, d_S);
    CUDA_CHECK(cudaDeviceSynchronize());

    // STEP 5: copy the full score matrix back to the host.
    std::vector<float> scores_host((size_t)B * N);
    CUDA_CHECK(cudaMemcpy(scores_host.data(), d_S,
                          (size_t)B * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // STEP 6: CPU top-k over approximate scores.
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

    // STEP 7: free device memory.
    CUDA_CHECK(cudaFree(d_Xq));
    CUDA_CHECK(cudaFree(d_X_scale));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_S));

    // STEP 8: populate and return SearchResult.
    auto t1 = std::chrono::high_resolution_clock::now();
    result.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

}
