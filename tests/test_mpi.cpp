// test_mpi - correctness test for distributed CUDAsearch.
//
// Run directly:
//   mpirun -np 4 ./test_mpi
//
// Or through CTest after configuring with USE_MPI=ON:
//   ctest -R test_mpi

#include "core/cpu_search.hpp"
#include "io/fvecs_loader.hpp"
#include "mpi/distributed_search.hpp"

#ifdef HAVE_CUDA
#include "cuda/gpu_search.cuh"
#endif

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
    int N = 4096;
    int B = 64;
    int d = 128;
    int k = 10;
    std::string kernel;
};

Config parse_args(int argc, char* argv[])
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* s) { return std::strcmp(argv[i], s) == 0; };
        auto next_int = [&]() { return std::atoi(argv[++i]); };
        auto next_str = [&]() -> std::string { return argv[++i]; };

        if (is("--N")) cfg.N = next_int();
        else if (is("--B")) cfg.B = next_int();
        else if (is("--d")) cfg.d = next_int();
        else if (is("--k")) cfg.k = next_int();
        else if (is("--kernel")) cfg.kernel = next_str();
        else {
            throw std::invalid_argument(std::string("unknown option: ") + argv[i]);
        }
    }
    return cfg;
}

bool is_int8_kernel(const std::string& kernel)
{
    return kernel == "int8" || kernel == "int8_tiled";
}

bool kernel_supported(const std::string& kernel)
{
    if (kernel == "cpu") return true;

#ifdef HAVE_CUDA
    return kernel == "naive" ||
           kernel == "tiled" ||
           kernel == "int8" ||
           kernel == "int8_tiled";
#else
    return false;
#endif
}

std::vector<std::string> kernels_to_test(const Config& cfg)
{
    if (!cfg.kernel.empty()) {
        return {cfg.kernel};
    }

#ifdef HAVE_CUDA
    return {"tiled", "int8_tiled"};
#else
    return {"cpu"};
#endif
}

core::SearchResult reference_search(const std::string& kernel,
                                    const io::Dataset& ds,
                                    int B, int k)
{
    if (kernel == "cpu") {
        return core::cpu_search(ds.base.data(), ds.n_base, ds.dim,
                                ds.queries.data(), B, k);
    }

#ifdef HAVE_CUDA
    if (kernel == "naive") {
        return core::gpu_search_naive(ds.base.data(), ds.n_base, ds.dim,
                                      ds.queries.data(), B, k);
    }
    if (kernel == "tiled") {
        return core::gpu_search_tiled(ds.base.data(), ds.n_base, ds.dim,
                                      ds.queries.data(), B, k);
    }
    if (kernel == "int8") {
        return core::gpu_search_int8(ds.base.data(), ds.n_base, ds.dim,
                                     ds.queries.data(), B, k);
    }
    if (kernel == "int8_tiled") {
        return core::gpu_search_int8_tiled(ds.base.data(), ds.n_base, ds.dim,
                                           ds.queries.data(), B, k);
    }
#endif

    throw std::invalid_argument("unsupported test kernel: " + kernel);
}

bool run_one_kernel(const std::string& kernel,
                    const Config& cfg,
                    int rank,
                    int world_size)
{
    if (!kernel_supported(kernel)) {
        throw std::invalid_argument("unsupported test kernel: " + kernel);
    }

    io::Dataset ds;
    core::SearchResult reference;

    if (rank == 0) {
        ds = io::make_random_dataset(cfg.N, cfg.B, cfg.d, cfg.k, 123);
        reference = reference_search(kernel, ds, cfg.B, cfg.k);
    }

    core::DistributedSearchConfig dist_cfg;
    dist_cfg.kernel = kernel;
    dist_cfg.root = 0;

    core::DistributedSearchIndex index(rank == 0 ? ds.base.data() : nullptr,
                                       rank == 0 ? ds.n_base : 0,
                                       rank == 0 ? ds.dim : 0,
                                       dist_cfg,
                                       MPI_COMM_WORLD);

    core::DistributedSearchResult distributed =
        index.search(rank == 0 ? ds.queries.data() : nullptr,
                     rank == 0 ? cfg.B : 0,
                     rank == 0 ? cfg.k : 0);

    bool ok = true;
    if (rank == 0) {
        const float match_recall =
            core::recall_at_k(distributed.result, reference.indices,
                              cfg.k, cfg.B, cfg.k);
        const float gt_recall =
            core::recall_at_k(distributed.result, ds.gt,
                              ds.gt_k, cfg.B, cfg.k);

        const float match_threshold = 0.999f;
        const float gt_threshold = is_int8_kernel(kernel) ? 0.98f : 0.999f;

        ok = match_recall >= match_threshold && gt_recall >= gt_threshold;

        std::printf("[%s] ranks=%d kernel=%s match_recall=%.4f "
                    "gt_recall=%.4f total_ms=%.2f\n",
                    ok ? "PASS" : "FAIL",
                    world_size,
                    kernel.c_str(),
                    match_recall,
                    gt_recall,
                    distributed.timing.total_ms);
    }

    int failed = ok ? 0 : 1;
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return failed == 0;
}

} // namespace

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int failed = 0;
    try {
        Config cfg = parse_args(argc, argv);
        if (cfg.N <= 0 || cfg.B <= 0 || cfg.d <= 0 ||
            cfg.k <= 0 || cfg.k > cfg.N) {
            throw std::invalid_argument("invalid synthetic test dimensions");
        }

        for (const std::string& kernel : kernels_to_test(cfg)) {
            if (!run_one_kernel(kernel, cfg, rank, world_size)) {
                failed = 1;
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rank %d] %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("%s - test_mpi with %d rank(s)\n",
                    any_failed ? "FAILED" : "PASSED",
                    world_size);
    }

    MPI_Finalize();
    return any_failed;
}
