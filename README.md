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

The INT8 tiled kernel quantizes the database once at index-build time and stores it transposed for coalesced reads, giving a **1.10× speedup over FP32 tiled** on SIFT1M and **1.49× on GIST1M** (d=960), with a small recall drop from quantization.

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

**GPU benchmark (choose `--kernel naive | tiled | int8 | int8_tiled`):**
```bash
./build/bench --data ./data/sift1m --dataset sift1m --kernel tiled --k 10 --batch 100 --trials 5
```

**MPI benchmark (4 ranks, SIFT1M):**
```bash
mpirun -np 4 ./build-mpi/mpi_bench \
  --data ./data/sift1m --dataset sift1m --kernel tiled \
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
