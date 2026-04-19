# CUDAsearch

Distributed GPU-accelerated dense vector similarity search.  
CME 213 Final Project — Ava Kouhana · Ethan Cohen · Stanford Spring 2026.

## What it does

Given a large pre-computed embedding database and one or more query vectors, CUDAsearch returns the **top-k maximum inner-product (MIPS)** results. This primitive drives semantic search, RAG retrieval, and recommendation systems.

The system is built in three layers, introduced progressively across milestones:

| Layer |
|---|
| CPU baseline (OpenMP) |
| CUDA kernels (naive + tiled + INT8) |
| MPI sharding across multiple GPUs |

## Datasets

| Dataset | Vectors | Dim | Size |
|---|---|---|---|
| SIFT1M | 1 M | 128 | ~500 MB |
| GIST1M | 1 M | 960 | ~3.6 GB |

Download:
```bash
bash scripts/download_datasets.sh --data ./data --only sift1m
bash scripts/download_datasets.sh --data ./data --only gist1m
```

## Build

Requirements: CMake ≥ 3.18, a C++17 compiler, (optionally) CUDA ≥ 11, OpenMPI.

```bash
cmake -B build -DUSE_OPENMP=ON
cmake --build build -j$(nproc)
```

With CUDA (M3+):
```bash
cmake -B build -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j$(nproc)
```

## Run

**Correctness test (synthetic, no data required):**
```bash
./build/test_recall
```

**Correctness test on real data:**
```bash
./build/test_recall --data ./data sift1m
```

**Benchmark (synthetic):**
```bash
./build/bench --synthetic --syn-N 1000000 --syn-d 128 --k 10 --batch 64
```

**Benchmark (SIFT1M):**
```bash
./build/bench --data ./data --dataset sift1m --k 10 --batch 100 --trials 5
```

**CSV output (for scripting):**
```bash
./build/bench --synthetic --csv
```

## Project structure

```
cudasearch/
  src/
    io/           # Dataset loaders (.fvecs / .ivecs / .bvecs)
    core/         # CPU baseline + recall evaluation
    mpi/          # MPI sharding and distributed top-k merge
  bench/          # Benchmark driver
  tests/          # Recall correctness tests
  scripts/        # Dataset download scripts
```

