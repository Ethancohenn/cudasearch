# CUDAsearch

Distributed GPU-accelerated dense vector similarity search.  
CME 213 Final Project — Ava Kouhana · Ethan Cohen · Stanford Spring 2026.

## What it does

Given a large pre-computed embedding database and one or more query vectors, CUDAsearch returns the **top-k maximum inner-product (MIPS)** results. This primitive drives semantic search, RAG retrieval, and recommendation systems.

The system is built in three layers:

| Layer |
|---|
| CPU baseline (OpenMP) |
| CUDA kernels (naive + tiled + row-major INT8 + tiled INT8) |
| MPI row sharding across multiple GPUs |

## Milestones

Course milestone writeups are in [`docs/milestones`](docs/milestones):

- [Milestone 1](docs/milestones/milestone-1.pdf)
- [Milestone 2](docs/milestones/milestone-2.pdf)
- [Milestone 3](docs/milestones/milestone-3.pdf)
- [Milestone 4](docs/milestones/milestone-4.pdf)

## Results

Benchmarks run on `hpcc-gpu-5-1` (Quadro RTX 6000, 16 CPU cores). Unless noted: SIFT1M, `N=1,000,000`, `B=100`, `k=10`.

**Kernel comparison (SIFT1M):**

| Kernel | Latency | QPS | Recall@10 | Speedup vs CPU-16 |
|---|---:|---:|---:|---:|
| CPU-16 (OpenMP) | 1108 ms | 90.3 | 0.9890 | 1.00× |
| CUDA tiled | 772 ms | 129.5 | 0.9900 | 1.43× |
| CUDA INT8 tiled | 699 ms | 143.0 | 0.9630 | 1.58× |
| MPI tiled (4 ranks) | 246 ms | 407.1 | 0.9890 | 4.50× |
| MPI INT8 tiled (4 ranks) | 222 ms | 450.2 | 0.9630 | 4.99× |
| MPI tiled + GPU top-k (4 ranks) | 31 ms | 3245.7 | 0.9890 | 35.96× |

The INT8 tiled kernel quantizes the database once at index-build time and stores it transposed for coalesced reads, giving a **1.10× speedup over FP32 tiled** on SIFT1M and **1.49× on GIST1M** (d=960), with a small recall drop from quantization.

The `tiled_topk` variant keeps top-k selection on the GPU. It still computes the
same FP32 tiled score matrix, but copies only `B*k` result pairs back to the host
instead of the full `B*N` score matrix. On SIFT1M at 4 ranks this reduces
latency from **245.63 ms to 30.81 ms** versus the original FP32 tiled MPI path.

**MPI strong scaling (FP32 tiled, fixed N=1M):**

| Dataset | Ranks | Latency | QPS | Recall@10 | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| SIFT1M | 1 | 743 ms | 134.6 | 0.9900 | 1.00× | 100% |
| SIFT1M | 2 | 473 ms | 211.6 | 0.9890 | 1.57× | 78.6% |
| SIFT1M | 4 | 246 ms | 407.1 | 0.9890 | 3.03× | 75.6% |
| GIST1M | 1 | 1482 ms | 67.5 | 0.3560 | 1.00× | 100% |
| GIST1M | 2 | 890 ms | 112.4 | 0.3560 | 1.67× | 83.3% |
| GIST1M | 4 | 386 ms | 259.2 | 0.3560 | 3.84× | 96.1% |

GIST1M (d=960) scales better than SIFT1M (d=128) because larger dot products improve the compute-to-communication ratio. The low GIST1M recall is consistent across all kernels and reflects a mismatch between our L2-normalized inner-product objective and the dataset's reference ground truth.

**Nsight Systems profile (FP32 tiled, SIFT1M, single search call):**

| Phase | 1 rank (ms) | 4 ranks, per rank (ms) | Scaling |
|---|---:|---:|---:|
| `gpu_search_tiled_kernel` (GPU) | 21.87 | 5.47 | 4.00× |
| D2H memcpy (B×N score matrix) | 41.21 | 11.39 | 3.62× |
| H2D memcpy (X shard, per call) | 46.92 | 12.18 | 3.85× |
| `cudaLaunchKernel` | 0.20 | 0.22 | 0.91× |
| GPU/CUDA-API subtotal | ~110 | ~29 | 3.79× |
| Host top-k (CSV `local_ms` − subtotal) | ~633 | ~216 | 2.93× |
| End-to-end `local_ms` (CSV, 5-trial mean) | 743 | 245 | 3.03× |

The GPU kernel itself is only ~22 ms — ~92% of `local_ms` at 1 rank is the host-side `nth_element` top-k over the B×N score matrix that the FP32 tiled path copies back D2H every call. GPU work scales near-ideally (3.6–4.0×); host top-k scales 2.93× because each rank still sorts B rows serially. `cudaLaunchKernel` (~0.2 ms) and the per-batch query H2D (~3 µs) are not material. This profile motivated the `tiled_topk` path above, which keeps top-k selection on the GPU and copies back only `B*k` result pairs. The FP32 `tiled_topk` path still materializes the score matrix on device and re-uploads `X` per call; the INT8 tiled path already uses a persistent device-side index that amortizes the X shard H2D.

**Nsight Systems profile (FP32 `tiled_topk`, SIFT1M, single search call):**

| Phase | 1 rank (ms) | 4 ranks, per rank (ms) | Scaling |
|---|---:|---:|---:|
| `gpu_search_tiled_kernel` (GPU) | 28.82 | 5.93 | 4.86× |
| `gpu_topk_kernel` (GPU) | 13.48 | 3.82 | 3.53× |
| H2D memcpy (X shard, per call) | 47.36 | 12.29 | 3.85× |
| D2H memcpy (B·k result pairs) | 0.002 | 0.001 | — |
| GPU subtotal | 89.66 | 22.05 | 4.07× |
| End-to-end `local_ms` (CSV, 5-trial mean) | 107.82 | 30.58 | 3.53× |

With host top-k removed, the breakdown confirms the limiter is the local GPU search path itself: the per-call FP32 **X upload is now the single largest GPU-side cost** — 47.36 ms (≈49% of local) at 1 rank and 12.29 ms (≈40%) at 4 ranks — exceeding the score kernel. D2H falls to ~2 µs (the 400 MB B×N score matrix shrinks to ~8 KB of `B*k` pairs). Making the FP32 database persistent on the GPU, as `int8_tiled` already does, would remove that upload and is the highest-impact remaining optimization. 

**Result files:**

| File | Contents |
|---|---|
| `results/cpu_scaling_sift1m.csv` | SIFT1M OpenMP CPU scaling |
| `results/cpu_scaling_gist1m.csv` | GIST1M OpenMP CPU scaling |
| `results/sift1m_comparison.csv` | SIFT1M CPU/GPU kernel comparison |
| `results/gist1m_comparison.csv` | GIST1M CPU/GPU kernel comparison |
| `results/synthetic_kernel_check.csv` | Synthetic correctness/performance check |
| `results/mpi/sift1m_strong_scaling.csv` | SIFT1M MPI FP32 tiled strong scaling |
| `results/mpi/sift1m_weak_scaling.csv` | SIFT1M MPI FP32 tiled weak scaling |
| `results/mpi/gist1m_strong_scaling.csv` | GIST1M MPI FP32 tiled strong scaling |
| `results/mpi/sift1m_int8_tiled_strong_scaling.csv` | SIFT1M MPI INT8 tiled strong scaling |
| `results/mpi/sift1m_tiled_topk_strong_scaling.csv` | SIFT1M MPI FP32 tiled + GPU top-k strong scaling |
| `results/mpi/sift1m_tiled_topk_weak_scaling.csv` | SIFT1M MPI FP32 tiled + GPU top-k weak scaling |
| `results/mpi/gist1m_tiled_topk_strong_scaling.csv` | GIST1M MPI FP32 tiled + GPU top-k strong scaling |
| `results/nsys/sift1m_tiled_1r_*.nsys-rep` | Nsight Systems profile, FP32 tiled, 1 rank |
| `results/nsys/sift1m_tiled_4r_*.nsys-rep` | Nsight Systems profile, FP32 tiled, 4 ranks |

## Datasets

| Dataset | Vectors | Dim | Size |
|---|---|---|---|
| SIFT1M | 1 M | 128 | ~500 MB |
| GIST1M | 1 M | 960 | ~3.6 GB |

```bash
bash scripts/download_datasets.sh --data ./data --only sift1m
bash scripts/download_datasets.sh --data ./data --only gist1m
```

## Build

Requirements: CMake ≥ 3.18, C++17, (optionally) CUDA ≥ 11, OpenMPI.

```bash
cmake -B build -DUSE_OPENMP=ON
cmake --build build -j$(nproc)
```

With CUDA (Quadro RTX 6000, sm_75):
```bash
cmake -B build -DUSE_CUDA=ON -DUSE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j$(nproc)
```

With CUDA and MPI:
```bash
cmake -B build-mpi -DUSE_MPI=ON -DUSE_CUDA=ON -DUSE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-mpi -j$(nproc)
```

## Run

**CPU benchmark (SIFT1M):**
```bash
./build/bench --data ./data/sift1m --dataset sift1m --k 10 --batch 100 --trials 5
```

**GPU benchmark (choose `--kernel naive | tiled | tiled_topk | int8 | int8_tiled`):**
```bash
./build/bench --data ./data/sift1m --dataset sift1m --kernel tiled_topk --k 10 --batch 100 --trials 5
```

**MPI benchmark (4 ranks, SIFT1M):**
```bash
mpirun -np 4 ./build-mpi/mpi_bench \
  --data ./data/sift1m --dataset sift1m --kernel tiled_topk \
  --n 1000000 --batch 100 --k 10 --trials 2
```

**Correctness tests (MPI):**
```bash
ctest --test-dir build-mpi -R test_mpi --output-on-failure
```

The SLURM helper runs the full MPI scaling sweep and writes CSVs to `results/mpi/`:
```bash
sbatch scripts/run_mpi_bench.sh
```

**Nsight Systems profiling (1 rank and 4 ranks):**
```bash
bash scripts/run_nsys_profile.sh \
  --data ./data/sift1m --dataset sift1m --kernel tiled --ranks 1

bash scripts/run_nsys_profile.sh \
  --data ./data/sift1m --dataset sift1m --kernel tiled --ranks 4
```

To profile the GPU top-k path instead, pass `--kernel tiled_topk`. 
```bash
bash scripts/run_nsys_profile.sh \
  --data ./data/sift1m --dataset sift1m --kernel tiled_topk --ranks 4
```

Open the generated `.qdrep` files in Nsight Systems and capture one screenshot
from each timeline. For the Milestone 4 report, focus on:

- Whether the 4-rank run remains dominated by GPU kernel time or shifts toward MPI gather/merge overhead
- Whether communication is concentrated at batch boundaries, since the current implementation uses bulk-synchronous `MPI_Bcast` and `MPI_Gatherv`
- Whether rank 0 shows extra host-side work during final merge of local top-k candidates

The current benchmark CSVs already suggest the expected trend: on SIFT1M FP32 tiled,
local search still dominates, but gather cost increases from `0.04 ms` at 1 rank to
`15.63 ms` at 4 ranks, making root-side communication/merge the main scaling bottleneck.

## Project structure

```
cudasearch/
  src/
    io/           # Dataset loaders (.fvecs / .ivecs / .bvecs)
    core/         # CPU baseline + recall evaluation
    cuda/         # CUDA naive, tiled, INT8, and INT8 tiled kernels
    mpi/          # MPI row-sharded distributed search
  bench/          # Single-node and MPI benchmark drivers
  tests/          # Recall correctness tests (single-GPU + MPI)
  scripts/        # Dataset download and cluster benchmark scripts
  results/        # Benchmark CSVs
    mpi/          # Distributed multi-GPU benchmark CSVs
    archive/      # Older/intermediate benchmark CSVs
  docs/
    milestones/   # Course milestone PDFs
```
