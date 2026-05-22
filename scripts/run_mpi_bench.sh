#!/usr/bin/env bash
#
# SLURM script for Milestone 4 MPI scaling runs.
#
# Submit:
#   sbatch scripts/run_mpi_bench.sh
#
# Useful overrides:
#   KERNEL=int8_tiled TRIALS=3 sbatch scripts/run_mpi_bench.sh
#   DATASETS="sift1m" sbatch scripts/run_mpi_bench.sh
#   WEAK_DATASETS="sift1m gist1m" sbatch scripts/run_mpi_bench.sh
#   MODULES="cuda/11.8 openmpi" sbatch scripts/run_mpi_bench.sh
#   LAUNCHER=srun sbatch scripts/run_mpi_bench.sh
#   DO_BUILD=0 sbatch scripts/run_mpi_bench.sh
#   RUN_SYNTHETIC=1 sbatch scripts/run_mpi_bench.sh
#
# If your GPU partition has one GPU per node, change the SBATCH resource lines
# to something like: --nodes=4, --ntasks-per-node=1, --gres=gpu:1.

#SBATCH --job-name=cudasearch-mpi
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=results/mpi_%j.out
#SBATCH --error=results/mpi_%j.err

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUILD_DIR="${BUILD_DIR:-build-mpi}"
RESULT_DIR="${RESULT_DIR:-results/mpi}"
DATA_ROOT="${DATA_ROOT:-./data}"
DATASETS="${DATASETS:-sift1m gist1m}"
WEAK_DATASETS="${WEAK_DATASETS:-sift1m}"
KERNEL="${KERNEL:-tiled}"
TRIALS="${TRIALS:-2}"
BATCH="${BATCH:-100}"
K="${K:-10}"
DIM="${DIM:-128}"
STRONG_N="${STRONG_N:-1000000}"
WEAK_N_PER_RANK="${WEAK_N_PER_RANK:-250000}"
RANKS_LIST="${RANKS_LIST:-1 2 4}"
RUN_REAL="${RUN_REAL:-1}"
RUN_SYNTHETIC="${RUN_SYNTHETIC:-0}"
LAUNCHER="${LAUNCHER:-mpirun}"
CUDA_ARCH="${CUDA_ARCH:-75}"
DO_BUILD="${DO_BUILD:-1}"

mkdir -p "$RESULT_DIR" results

if command -v module >/dev/null 2>&1; then
    module purge
    for mod in ${MODULES:-cuda openmpi cmake}; do
        module load "$mod" || true
    done
fi

if [[ "$DO_BUILD" == "1" ]]; then
    cmake -B "$BUILD_DIR" \
        -DUSE_MPI=ON \
        -DUSE_CUDA=ON \
        -DUSE_OPENMP=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    cmake --build "$BUILD_DIR" -j
fi

run_mpi() {
    local ranks="$1"
    shift

    if [[ "$LAUNCHER" == "srun" ]]; then
        srun -n "$ranks" "$@"
    else
        mpirun -np "$ranks" "$@"
    fi
}

HEADER="dataset,kernel,ranks,N,d,k,B,mean_ms,min_ms,bcast_ms,local_ms,gather_ms,merge_ms,setup_ms,qps,recall"

run_real_strong() {
    local csv="$RESULT_DIR/strong_scaling.csv"
    echo "$HEADER" > "$csv"

    echo "Running real-data strong scaling: datasets=[$DATASETS], fixed N=$STRONG_N"
    for dataset in $DATASETS; do
        local data_dir="$DATA_ROOT/$dataset"
        if [[ ! -f "$data_dir/${dataset}_base.fvecs" ]]; then
            echo "Missing $data_dir/${dataset}_base.fvecs"
            echo "Download first: bash scripts/download_datasets.sh --data $DATA_ROOT --only $dataset"
            exit 1
        fi

        for ranks in $RANKS_LIST; do
            echo "  dataset=$dataset ranks=$ranks"
            run_mpi "$ranks" "$BUILD_DIR/mpi_bench" \
                --data "$data_dir" \
                --dataset "$dataset" \
                --n "$STRONG_N" \
                --kernel "$KERNEL" \
                --k "$K" \
                --batch "$BATCH" \
                --trials "$TRIALS" \
                --csv >> "$csv"
        done
    done

    echo "Wrote $csv"
}

run_real_weak() {
    local csv="$RESULT_DIR/weak_scaling.csv"
    echo "$HEADER" > "$csv"

    echo "Running real-data weak scaling: datasets=[$WEAK_DATASETS], N_per_rank=$WEAK_N_PER_RANK"
    for dataset in $WEAK_DATASETS; do
        local data_dir="$DATA_ROOT/$dataset"
        if [[ ! -f "$data_dir/${dataset}_base.fvecs" ]]; then
            echo "Missing $data_dir/${dataset}_base.fvecs"
            echo "Download first: bash scripts/download_datasets.sh --data $DATA_ROOT --only $dataset"
            exit 1
        fi

        for ranks in $RANKS_LIST; do
            local N=$((ranks * WEAK_N_PER_RANK))
            echo "  dataset=$dataset ranks=$ranks N=$N"
            run_mpi "$ranks" "$BUILD_DIR/mpi_bench" \
                --data "$data_dir" \
                --dataset "$dataset" \
                --n "$N" \
                --kernel "$KERNEL" \
                --k "$K" \
                --batch "$BATCH" \
                --trials "$TRIALS" \
                --csv >> "$csv"
        done
    done

    echo "Wrote $csv"
}

run_synthetic_sweeps() {
    local strong_csv="$RESULT_DIR/synthetic_strong_scaling.csv"
    local weak_csv="$RESULT_DIR/synthetic_weak_scaling.csv"
    echo "$HEADER" > "$strong_csv"
    echo "$HEADER" > "$weak_csv"

    echo "Running synthetic strong scaling: fixed N=$STRONG_N"
    for ranks in $RANKS_LIST; do
        echo "  ranks=$ranks"
        run_mpi "$ranks" "$BUILD_DIR/mpi_bench" \
            --synthetic \
            --syn-N "$STRONG_N" \
            --syn-B "$BATCH" \
            --syn-d "$DIM" \
            --kernel "$KERNEL" \
            --k "$K" \
            --batch "$BATCH" \
            --trials "$TRIALS" \
            --csv >> "$strong_csv"
    done

    echo "Running synthetic weak scaling: N_per_rank=$WEAK_N_PER_RANK"
    for ranks in $RANKS_LIST; do
        local N=$((ranks * WEAK_N_PER_RANK))
        echo "  ranks=$ranks N=$N"
        run_mpi "$ranks" "$BUILD_DIR/mpi_bench" \
            --synthetic \
            --syn-N "$N" \
            --syn-B "$BATCH" \
            --syn-d "$DIM" \
            --kernel "$KERNEL" \
            --k "$K" \
            --batch "$BATCH" \
            --trials "$TRIALS" \
            --csv >> "$weak_csv"
    done

    echo "Wrote $strong_csv"
    echo "Wrote $weak_csv"
}

if [[ "$RUN_REAL" == "1" ]]; then
    run_real_strong
    run_real_weak
fi

if [[ "$RUN_SYNTHETIC" == "1" ]]; then
    run_synthetic_sweeps
fi
