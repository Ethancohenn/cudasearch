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

Final benchmarks were run on the same `gpu-turing` node (`hpcc-gpu-5-1`) with one Quadro RTX 6000 GPU and 16 allocated CPU cores:

```bash
srun --partition=gpu-turing --nodelist=hpcc-gpu-5-1 \
  --gres=gpu:1 --cpus-per-task=16 --mem=32G --pty bash
```

All runs use `N=1,000,000` base vectors, batch size `B=100`, and `k=10`.


**SIFT1M CPU scaling (OpenMP):**

| Threads | mean latency | QPS |
|---|---|---|
| 1  | 5472 ms | 18.3 |
| 2  | 2941 ms | 34.0 |
| 4  | 1634 ms | 61.2 |
| 8  | 1427 ms | 70.1 |
| 16 | 1108 ms | 90.3 |

Every CPU configuration returns **Recall@10 = 0.9890** against the SIFT1M reference ground truth. The best CPU baseline is the 16-thread run. OpenMP gives a 4.94× speedup from 1 to 16 threads, but scaling is sublinear, consistent with an exhaustive scan that becomes increasingly limited by shared memory bandwidth and threading overhead.

**SIFT1M GPU comparison:**

| Kernel | mean latency | QPS | Recall@10 | speedup vs CPU-16 |
|---|---:|---:|---:|---:|
| CPU-16 | 1108 ms | 90.3 | 0.9890 | 1.00× |
| CUDA naive | 1047 ms | 95.5 | 0.9900 | 1.06× |
| CUDA tiled | 772 ms | 129.5 | 0.9900 | 1.43× |
| CUDA INT8 | 1893 ms | 52.8 | 0.9630 | 0.59× |

The naive CUDA kernel assigns one thread to one full dot product. The tiled kernel uses 32×32 thread blocks and shared memory tiles, so each block computes a 32×32 tile of the score matrix. On SIFT1M, tiling gives a **1.36× speedup over naive CUDA**.

**GIST1M comparison:**

| Kernel | mean latency | QPS | Recall@10 | speedup vs CPU |
|---|---:|---:|---:|---:|
| CPU-16 | 7507 ms | 13.3 | 0.3560 | 1.00× |
| CUDA naive | 8926 ms | 11.2 | 0.3560 | 0.84× |
| CUDA tiled | 1542 ms | 64.9 | 0.3560 | 4.87× |
| CUDA INT8 | 6497 ms | 15.4 | 0.3540 | 1.16× |

GIST1M has dimension `d=960`, compared with `d=128` for SIFT1M. The longer dot products make shared-memory tiling much more valuable: the tiled kernel is **5.79× faster than naive CUDA** on GIST1M. The INT8 kernel is also more useful on GIST1M than on SIFT1M, improving over naive CUDA by **1.37×** with almost no additional recall loss. It is still much slower than tiled because this first INT8 path quantizes `X` inside each search call, converts INT8 values back to float in the kernel, and still computes top-k on the CPU. The low GIST1M recall is not a CPU/GPU correctness issue because all implementations agree; it likely reflects a mismatch between our L2-normalized inner-product objective and the dataset's reference ground truth.

**Synthetic kernel check:**

| Kernel | mean latency | QPS | Recall@10 |
|---|---:|---:|---:|
| CPU | 40.13 ms | 2491.7 | 1.0000 |
| CUDA naive | 102.85 ms | 972.3 | 1.0000 |
| CUDA tiled | 83.22 ms | 1201.6 | 1.0000 |
| CUDA INT8 | 120.21 ms | 831.9 | 0.9890 |

The synthetic check confirms that CPU, naive CUDA, and tiled CUDA match the generated ground truth exactly, while INT8 has a small expected recall drop from quantization.

Earlier CPU-only measurements were collected from a GPU allocation that did not explicitly reserve CPU cores. Those results are kept in `results/cpu_scaling.csv` for reference, but the final speedups above use the explicit 16-core same-node allocation.

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

With CUDA on the Quadro RTX 6000:
```bash
cmake -B build -DUSE_CUDA=ON -DUSE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=75
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

**Benchmark (SIFT1M, GPU kernels):**
```bash
./build/bench --data ./data/sift1m --dataset sift1m --kernel naive --k 10 --batch 100 --trials 5
./build/bench --data ./data/sift1m --dataset sift1m --kernel tiled --k 10 --batch 100 --trials 5
./build/bench --data ./data/sift1m --dataset sift1m --kernel int8 --k 10 --batch 100 --trials 5
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
    cuda/         # CUDA naive, tiled, and INT8 implementations
  bench/          # Benchmark driver
  tests/          # Recall correctness tests
  scripts/        # Dataset download scripts
```
