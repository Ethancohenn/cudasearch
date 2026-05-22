#pragma once

#include "core/cpu_search.hpp"

#include <mpi.h>

#include <memory>
#include <string>
#include <vector>

namespace core {

#ifdef HAVE_CUDA
class GpuInt8TiledIndex;
#endif

struct DistributedSearchConfig {
    // Must be the same on every rank.
    // Supported kernels: cpu, naive, tiled, int8, int8_tiled.
    std::string kernel = "cpu";
    int root = 0;
};

struct DistributedSearchTiming {
    double setup_ms = 0.0;
    double scatter_ms = 0.0;
    double index_build_ms = 0.0;

    double total_ms = 0.0;
    double query_bcast_ms = 0.0;
    double local_search_ms = 0.0;
    double candidate_gather_ms = 0.0;
    double merge_ms = 0.0;

    double communication_ms() const {
        return query_bcast_ms + candidate_gather_ms;
    }
};

struct DistributedSearchResult {
    // Filled only on config.root.
    SearchResult result;
    bool valid_on_this_rank = false;

    DistributedSearchTiming timing;
    int rank = 0;
    int world_size = 1;
    int local_N = 0;
    int global_offset = 0;
};

// Collective over comm. The root rank passes the full row-major database X;
// other ranks pass nullptr/0. The constructor scatters contiguous row shards.
class DistributedSearchIndex {
public:
    DistributedSearchIndex(const float* X_root, int N, int d,
                           const DistributedSearchConfig& config,
                           MPI_Comm comm = MPI_COMM_WORLD);
    ~DistributedSearchIndex();

    DistributedSearchIndex(const DistributedSearchIndex&) = delete;
    DistributedSearchIndex& operator=(const DistributedSearchIndex&) = delete;

    // Collective over the same communicator. Root passes the full query batch;
    // all ranks receive the same queries, search their local shard, and root
    // merges the gathered local top-k candidates.
    DistributedSearchResult search(const float* Q_root, int B, int k) const;

    const DistributedSearchTiming& setup_timing() const { return setup_; }
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    int local_size() const { return local_N_; }
    int global_offset() const { return global_offset_; }
    int dim() const { return d_; }
    int global_size() const { return N_; }

private:
    SearchResult local_search(const float* Q, int B, int k,
                              double& elapsed_ms) const;

    MPI_Comm comm_ = MPI_COMM_NULL;
    DistributedSearchConfig config_;
    int rank_ = 0;
    int world_size_ = 1;
    int root_ = 0;

    int N_ = 0;
    int d_ = 0;
    int local_N_ = 0;
    int global_offset_ = 0;

    std::vector<float> X_local_;
    DistributedSearchTiming setup_;

#ifdef HAVE_CUDA
    std::unique_ptr<GpuInt8TiledIndex> int8_index_;
#endif
};

} // namespace core
