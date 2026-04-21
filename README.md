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

## Results

We compare the CPU baseline against the a naive implementation of GPU kernel where each thread runs one dot product computation on the SIFT1M dataset (1M × 128 float32 vectors), running a batch of B=100 queries with top-k=10. 


**CPU (OpenMP):**

| Threads | mean latency | QPS |
|---|---|---|
| 1  | 4964 ms | 20.1 |
| 2  | 4296 ms | 23.3 |
| 4  | 4312 ms | 23.2 |
| 8  | 4401 ms | 22.7 |
| 16 | 4627 ms | 21.6 |


Every configuration returns **Recall@10 = 0.9890** against the dataset's reference ground truth. 

Throughput peaks at 2 threads and degrades past that. This is a symptom of a memory-bound computation as the computation has arithmetic intensity ~0.5 FLOP/byte (one multiply-add per 4-byte float read ), which, according to the roofline graph, sits far below the CPU's compute/bandwidth crossover point. In other words, two threads are already enough to consume the available memory bandwidth; 

Adding more threads does not add usable bandwidth, only contention. The CPU cannot be made faster on this workload without more memory bandwidth — motivating the need for a GPU implementation.

**GPU:**

| Kernel | mean latency | QPS | speedup vs CPU best |
|---|---|---|---|
| naive (v1) | 1044 ms | 96 | 4.1× |

The GPU completes the same batch in roughly a quarter of the CPU time (**3.94× speedup** on the same-node Quadro RTX 6000).

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

**Benchmark (SIFT1M, GPU naive kernel):**
```bash
./build/bench --data ./data/sift1m --dataset sift1m --kernel naive --k 10 --batch 100 --trials 5
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
    cuda/         # CUDA naive implementation
  bench/          # Benchmark driver
  tests/          # Recall correctness tests
  scripts/        # Dataset download scripts
```

