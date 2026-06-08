#!/usr/bin/env bash
#
# Lightweight Nsight Systems profiling helper for CUDAsearch MPI runs.
#
# Examples:
#   bash scripts/run_nsys_profile.sh --data ./data/sift1m --dataset sift1m --ranks 1
#   bash scripts/run_nsys_profile.sh --data ./data/sift1m --dataset sift1m --ranks 4
#   bash scripts/run_nsys_profile.sh --data ./data/sift1m --dataset sift1m --kernel tiled_topk --ranks 4
#
# Notes:
# - Run this on a machine with CUDA, MPI, and Nsight Systems installed.
# - Keep profiles short: use 1 trial and a modest batch size so the report
#   stays readable and cluster usage remains reasonable.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-mpi}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/nsys}"
LAUNCHER="${LAUNCHER:-mpirun}"
CUDA_ARCH="${CUDA_ARCH:-75}"

DATA_DIR=""
DATASET="sift1m"
KERNEL="tiled"
RANKS=1
N=1000000
BATCH=100
K=10
TRIALS=1
FORCE_BUILD=0

usage() {
    cat <<EOF
Usage:
  bash scripts/run_nsys_profile.sh --data <dir> [options]

Required:
  --data <dir>         Dataset directory, e.g. ./data/sift1m

Options:
  --dataset <name>     Dataset name (default: sift1m)
  --kernel <name>      Kernel: cpu, naive, tiled, tiled_topk, int8, int8_tiled
  --ranks <int>        MPI rank count (default: 1)
  --n <int>            Database size limit (default: 1000000)
  --batch <int>        Query batch size (default: 100)
  --k <int>            Top-k (default: 10)
  --trials <int>       Timed search calls inside mpi_bench (default: 1)
  --build              Reconfigure and rebuild build-mpi before profiling

Environment overrides:
  BUILD_DIR, OUT_DIR, LAUNCHER, CUDA_ARCH
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data) DATA_DIR="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --kernel) KERNEL="$2"; shift 2 ;;
        --ranks) RANKS="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --build) FORCE_BUILD=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$DATA_DIR" ]]; then
    echo "--data is required" >&2
    usage
    exit 1
fi

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys not found on PATH" >&2
    exit 1
fi

if ! command -v "$LAUNCHER" >/dev/null 2>&1; then
    echo "$LAUNCHER not found on PATH" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

if [[ "$FORCE_BUILD" == "1" || ! -x "$BUILD_DIR/mpi_bench" ]]; then
    cmake -B "$BUILD_DIR" \
        -DUSE_MPI=ON \
        -DUSE_CUDA=ON \
        -DUSE_OPENMP=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    cmake --build "$BUILD_DIR" -j
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_NAME="${DATASET}_${KERNEL}_${RANKS}r_${STAMP}"
OUT_BASE="$OUT_DIR/$BASE_NAME"

echo "Writing profile to:"
echo "  $OUT_BASE.qdrep"
echo

set -x
nsys profile \
    --output "$OUT_BASE" \
    --trace cuda,nvtx,osrt \
    --sample none \
    "$LAUNCHER" -np "$RANKS" \
    "$BUILD_DIR/mpi_bench" \
    --data "$DATA_DIR" \
    --dataset "$DATASET" \
    --kernel "$KERNEL" \
    --n "$N" \
    --batch "$BATCH" \
    --k "$K" \
    --trials "$TRIALS"
set +x

echo
echo "Generated:"
echo "  $OUT_BASE.qdrep"
echo
echo "Optional summary commands:"
echo "  nsys stats --report cuda_api_sum,osrt_sum $OUT_BASE.qdrep"
echo "  nsys-ui $OUT_BASE.qdrep"
