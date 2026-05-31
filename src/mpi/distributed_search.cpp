#include "mpi/distributed_search.hpp"

#include "core/cpu_search.hpp"

#ifdef HAVE_CUDA
#include "cuda/gpu_search.cuh"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <climits>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace core {
namespace {

double elapsed_ms(double start)
{
    return (MPI_Wtime() - start) * 1000.0;
}

double max_ms(double local_ms, MPI_Comm comm)
{
    double global_ms = 0.0;
    MPI_Allreduce(&local_ms, &global_ms, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global_ms;
}

int mpi_count(long long count, const char* label)
{
    if (count < 0 || count > INT_MAX) {
        throw std::overflow_error(std::string(label) + " exceeds MPI int count");
    }
    return static_cast<int>(count);
}

void row_partition(int N, int P, std::vector<int>& rows, std::vector<int>& offsets)
{
    rows.assign(P, N / P);
    offsets.assign(P, 0);

    for (int r = 0; r < N % P; ++r) {
        ++rows[r];
    }
    for (int r = 1; r < P; ++r) {
        offsets[r] = offsets[r - 1] + rows[r - 1];
    }
}

bool is_cuda_kernel(const std::string& kernel)
{
    return kernel == "naive" ||
           kernel == "tiled" ||
           kernel == "tiled_topk" ||
           kernel == "int8" ||
           kernel == "int8_tiled";
}

void validate_kernel(const std::string& kernel)
{
    if (kernel == "cpu") return;

#ifdef HAVE_CUDA
    if (is_cuda_kernel(kernel)) return;
#endif

    if (is_cuda_kernel(kernel)) {
        throw std::invalid_argument(
            "CUDA kernel requested, but this target was not built with CUDA");
    }
    throw std::invalid_argument("unknown distributed kernel: " + kernel);
}

#ifdef HAVE_CUDA
void cuda_check(cudaError_t err, const char* expr, const char* file, int line)
{
    if (err != cudaSuccess) {
        char msg[512];
        std::snprintf(msg, sizeof(msg), "CUDA error at %s:%d for %s: %s",
                      file, line, expr, cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }
}

#define CUDA_CHECK_MPI(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

void select_gpu_for_local_rank(MPI_Comm comm)
{
    int n_devices = 0;
    CUDA_CHECK_MPI(cudaGetDeviceCount(&n_devices));
    if (n_devices <= 0) {
        throw std::runtime_error("CUDA kernel requested, but no GPU was found");
    }

    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);

    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);

    CUDA_CHECK_MPI(cudaSetDevice(local_rank % n_devices));
}
#endif

struct Candidate {
    float score;
    int index;
};

bool better(const Candidate& a, const Candidate& b)
{
    if (a.score != b.score) return a.score > b.score;
    return a.index < b.index;
}

SearchResult merge_topk_on_root(const std::vector<int>& all_indices,
                                const std::vector<float>& all_scores,
                                const std::vector<int>& local_ks,
                                const std::vector<int>& displs,
                                int B, int k)
{
    SearchResult out;
    out.indices.resize(static_cast<size_t>(B) * k);
    out.scores.resize(static_cast<size_t>(B) * k);

    int candidates_per_query = 0;
    for (int local_k : local_ks) {
        candidates_per_query += local_k;
    }
    if (candidates_per_query < k) {
        throw std::runtime_error("not enough local candidates for global top-k");
    }

    std::vector<Candidate> candidates;
    candidates.reserve(candidates_per_query);

    for (int b = 0; b < B; ++b) {
        candidates.clear();
        for (size_t r = 0; r < local_ks.size(); ++r) {
            const int local_k = local_ks[r];
            const int base = displs[r] + b * local_k;
            for (int j = 0; j < local_k; ++j) {
                candidates.push_back({all_scores[base + j],
                                      all_indices[base + j]});
            }
        }

        if (static_cast<int>(candidates.size()) > k) {
            std::nth_element(candidates.begin(), candidates.begin() + k,
                             candidates.end(), better);
        }
        std::sort(candidates.begin(), candidates.begin() + k, better);

        int* out_idx = out.indices.data() + static_cast<size_t>(b) * k;
        float* out_score = out.scores.data() + static_cast<size_t>(b) * k;
        for (int j = 0; j < k; ++j) {
            out_idx[j] = candidates[j].index;
            out_score[j] = candidates[j].score;
        }
    }

    return out;
}

} // namespace

DistributedSearchIndex::DistributedSearchIndex(
    const float* X_root, int N, int d,
    const DistributedSearchConfig& config,
    MPI_Comm comm)
    : config_(config), root_(config.root)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        throw std::runtime_error("DistributedSearchIndex requires MPI_Init first");
    }

    const double setup_start = MPI_Wtime();

    MPI_Comm_dup(comm, &comm_);
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &world_size_);

    if (root_ < 0 || root_ >= world_size_) {
        throw std::invalid_argument("MPI root rank is outside the communicator");
    }

    int meta[2] = {0, 0};
    if (rank_ == root_) {
        if (!X_root || N <= 0 || d <= 0) {
            throw std::invalid_argument("root must provide X, N > 0, and d > 0");
        }
        meta[0] = N;
        meta[1] = d;
    }
    MPI_Bcast(meta, 2, MPI_INT, root_, comm_);
    N_ = meta[0];
    d_ = meta[1];

    validate_kernel(config_.kernel);

#ifdef HAVE_CUDA
    if (is_cuda_kernel(config_.kernel)) {
        select_gpu_for_local_rank(comm_);
    }
#endif

    std::vector<int> rows;
    std::vector<int> offsets;
    row_partition(N_, world_size_, rows, offsets);
    local_N_ = rows[rank_];
    global_offset_ = offsets[rank_];

    std::vector<int> sendcounts;
    std::vector<int> displs;
    if (rank_ == root_) {
        sendcounts.resize(world_size_);
        displs.resize(world_size_);
        for (int r = 0; r < world_size_; ++r) {
            sendcounts[r] = mpi_count(static_cast<long long>(rows[r]) * d_,
                                      "scatter send count");
            displs[r] = mpi_count(static_cast<long long>(offsets[r]) * d_,
                                  "scatter displacement");
        }
    }

    X_local_.resize(static_cast<size_t>(local_N_) * d_);
    const int recv_count = mpi_count(static_cast<long long>(local_N_) * d_,
                                     "scatter receive count");

    MPI_Barrier(comm_);
    const double scatter_start = MPI_Wtime();
    MPI_Scatterv(rank_ == root_ ? X_root : nullptr,
                 rank_ == root_ ? sendcounts.data() : nullptr,
                 rank_ == root_ ? displs.data() : nullptr,
                 MPI_FLOAT,
                 X_local_.empty() ? nullptr : X_local_.data(),
                 recv_count,
                 MPI_FLOAT,
                 root_,
                 comm_);
    setup_.scatter_ms = max_ms(elapsed_ms(scatter_start), comm_);

#ifdef HAVE_CUDA
    if (config_.kernel == "int8_tiled") {
        double build_ms = 0.0;
        if (local_N_ > 0) {
            const double build_start = MPI_Wtime();
            int8_index_ = std::make_unique<GpuInt8TiledIndex>(
                X_local_.data(), local_N_, d_);
            build_ms = elapsed_ms(build_start);
        }
        setup_.index_build_ms = max_ms(build_ms, comm_);
    }
#endif

    MPI_Barrier(comm_);
    setup_.setup_ms = max_ms(elapsed_ms(setup_start), comm_);
}

DistributedSearchIndex::~DistributedSearchIndex()
{
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&comm_);
    }
}

SearchResult DistributedSearchIndex::local_search(const float* Q, int B, int k,
                                                  double& elapsed) const
{
    const int local_k = std::min(k, local_N_);
    if (local_k == 0) {
        elapsed = 0.0;
        return SearchResult{};
    }

    const double start = MPI_Wtime();
    SearchResult out;

    if (config_.kernel == "cpu") {
        out = cpu_search(X_local_.data(), local_N_, d_, Q, B, local_k);
    }
#ifdef HAVE_CUDA
    else if (config_.kernel == "naive") {
        out = gpu_search_naive(X_local_.data(), local_N_, d_, Q, B, local_k);
    } else if (config_.kernel == "tiled") {
        out = gpu_search_tiled(X_local_.data(), local_N_, d_, Q, B, local_k);
    } else if (config_.kernel == "tiled_topk") {
        out = gpu_search_tiled_topk(X_local_.data(), local_N_, d_, Q, B, local_k);
    } else if (config_.kernel == "int8") {
        out = gpu_search_int8(X_local_.data(), local_N_, d_, Q, B, local_k);
    } else if (config_.kernel == "int8_tiled") {
        out = int8_index_->search(Q, B, local_k);
    }
#endif
    else {
        throw std::invalid_argument("unknown local kernel: " + config_.kernel);
    }

    for (int& index : out.indices) {
        index += global_offset_;
    }

    elapsed = elapsed_ms(start);
    return out;
}

DistributedSearchResult DistributedSearchIndex::search(const float* Q_root,
                                                       int B, int k) const
{
    int query_meta[2] = {0, 0};
    if (rank_ == root_) {
        if (!Q_root || B <= 0 || k <= 0 || k > N_) {
            throw std::invalid_argument("root must provide Q, B > 0, and 0 < k <= N");
        }
        query_meta[0] = B;
        query_meta[1] = k;
    }
    MPI_Bcast(query_meta, 2, MPI_INT, root_, comm_);
    B = query_meta[0];
    k = query_meta[1];

#ifdef HAVE_CUDA
    if (config_.kernel == "int8_tiled" && int8_index_) {
        int8_index_->reserve_query_capacity(B);
    }
#endif

    DistributedSearchResult out;
    out.rank = rank_;
    out.world_size = world_size_;
    out.local_N = local_N_;
    out.global_offset = global_offset_;
    out.timing = setup_;

    MPI_Barrier(comm_);
    const double total_start = MPI_Wtime();

    std::vector<float> Q(static_cast<size_t>(B) * d_);
    if (rank_ == root_) {
        std::copy(Q_root, Q_root + static_cast<size_t>(B) * d_, Q.begin());
    }

    const double bcast_start = MPI_Wtime();
    MPI_Bcast(Q.data(),
              mpi_count(static_cast<long long>(B) * d_, "query broadcast count"),
              MPI_FLOAT,
              root_,
              comm_);
    const double bcast_ms = elapsed_ms(bcast_start);

    double search_ms = 0.0;
    SearchResult local = local_search(Q.data(), B, k, search_ms);
    const int local_k = std::min(k, local_N_);
    const int send_count = B * local_k;

    std::vector<int> local_ks;
    if (rank_ == root_) {
        local_ks.resize(world_size_);
    }

    const double gather_start = MPI_Wtime();
    MPI_Gather(&local_k, 1, MPI_INT,
               rank_ == root_ ? local_ks.data() : nullptr, 1, MPI_INT,
               root_, comm_);

    std::vector<int> recvcounts;
    std::vector<int> displs;
    int total_recv = 0;
    if (rank_ == root_) {
        recvcounts.resize(world_size_);
        displs.resize(world_size_);
        for (int r = 0; r < world_size_; ++r) {
            recvcounts[r] = B * local_ks[r];
            displs[r] = total_recv;
            total_recv += recvcounts[r];
        }
    }

    std::vector<int> all_indices(rank_ == root_ ? total_recv : 0);
    std::vector<float> all_scores(rank_ == root_ ? total_recv : 0);

    MPI_Gatherv(local.indices.empty() ? nullptr : local.indices.data(),
                send_count, MPI_INT,
                rank_ == root_ ? all_indices.data() : nullptr,
                rank_ == root_ ? recvcounts.data() : nullptr,
                rank_ == root_ ? displs.data() : nullptr,
                MPI_INT, root_, comm_);

    MPI_Gatherv(local.scores.empty() ? nullptr : local.scores.data(),
                send_count, MPI_FLOAT,
                rank_ == root_ ? all_scores.data() : nullptr,
                rank_ == root_ ? recvcounts.data() : nullptr,
                rank_ == root_ ? displs.data() : nullptr,
                MPI_FLOAT, root_, comm_);

    const double gather_ms = elapsed_ms(gather_start);

    double merge_ms = 0.0;
    if (rank_ == root_) {
        const double merge_start = MPI_Wtime();
        out.result = merge_topk_on_root(all_indices, all_scores,
                                        local_ks, displs, B, k);
        merge_ms = elapsed_ms(merge_start);
        out.valid_on_this_rank = true;
    }

    MPI_Barrier(comm_);
    const double total_ms = elapsed_ms(total_start);

    out.timing.query_bcast_ms = max_ms(bcast_ms, comm_);
    out.timing.local_search_ms = max_ms(search_ms, comm_);
    out.timing.candidate_gather_ms = max_ms(gather_ms, comm_);
    out.timing.merge_ms = max_ms(merge_ms, comm_);
    out.timing.total_ms = max_ms(total_ms, comm_);

    if (rank_ == root_) {
        out.result.wall_ms = out.timing.total_ms;
    }

    return out;
}

} // namespace core
