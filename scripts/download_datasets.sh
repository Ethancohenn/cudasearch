#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Download ANN benchmark datasets for CUDAsearch
#
#  Datasets:
#    SIFT1M  — 1M vectors, dim 128, float32   (~500 MB)
#              Standard MIPS benchmark, fast to load, excellent for development.
#    GIST1M  — 1M vectors, dim 960, float32   (~3.6 GB)
#              High-dimensional variant; stress-tests memory bandwidth.
#
#  Usage:
#    bash scripts/download_datasets.sh [--data <dir>] [--only sift1m|gist1m]
#
#  The script will:
#    1. Download the tar.gz archive to <dir>/
#    2. Extract it
#    3. Rename files to the cudasearch convention:
#         <name>_base.fvecs
#         <name>_query.fvecs
#         <name>_groundtruth.ivecs
#
#  Source: http://corpus-texmex.irisa.fr/  (ANN_SIFT1M, ANN_GIST1M)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DATA_DIR="./data"
ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data)  DATA_DIR="$2"; shift 2 ;;
        --only)  ONLY="$2";     shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$DATA_DIR"

# ─── Download helper ──────────────────────────────────────────────────────────
download_and_extract() {
    local name="$1"   # e.g. sift1m
    local url="$2"    # tar.gz URL
    local src_dir="$3"  # directory name inside the archive (e.g. sift)
    local src_base="$4"     # filename for base vectors inside src_dir
    local src_query="$5"
    local src_gt="$6"

    echo ""
    echo "── $name ────────────────────────────────────────────────────────────"

    local archive="$DATA_DIR/${name}.tar.gz"
    local dest_dir="$DATA_DIR/$name"

    if [[ -f "$dest_dir/${name}_base.fvecs" ]]; then
        echo "  Already downloaded. Skipping."
        return
    fi

    # Download
    if [[ ! -f "$archive" ]]; then
        echo "  Downloading from $url ..."
        if command -v wget &>/dev/null; then
            wget -q --show-progress -O "$archive" "$url"
        else
            curl -L --progress-bar -o "$archive" "$url"
        fi
    else
        echo "  Archive already present: $archive"
    fi

    # Extract
    echo "  Extracting ..."
    mkdir -p "$dest_dir"
    tar -xzf "$archive" -C "$DATA_DIR"

    # Rename to cudasearch convention
    echo "  Renaming files ..."
    mv "$DATA_DIR/$src_dir/$src_base"  "$dest_dir/${name}_base.fvecs"
    mv "$DATA_DIR/$src_dir/$src_query" "$dest_dir/${name}_query.fvecs"
    mv "$DATA_DIR/$src_dir/$src_gt"    "$dest_dir/${name}_groundtruth.ivecs"

    # Remove now-empty source directory
    rmdir "$DATA_DIR/$src_dir" 2>/dev/null || true

    echo "  Done → $dest_dir/"
    ls -lh "$dest_dir/"
}

# ─── SIFT1M ───────────────────────────────────────────────────────────────────
# Source: http://corpus-texmex.irisa.fr/
# 1,000,000 SIFT descriptors, dimension 128, float32
# 10,000 query vectors, ground truth: top-100 exact neighbours
if [[ -z "$ONLY" || "$ONLY" == "sift1m" ]]; then
    download_and_extract \
        "sift1m" \
        "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" \
        "sift" \
        "sift_base.fvecs" \
        "sift_query.fvecs" \
        "sift_groundtruth.ivecs"
fi

# ─── GIST1M ───────────────────────────────────────────────────────────────────
# Source: http://corpus-texmex.irisa.fr/
# 1,000,000 GIST descriptors, dimension 960, float32
# 1,000 query vectors, ground truth: top-100 exact neighbours
if [[ -z "$ONLY" || "$ONLY" == "gist1m" ]]; then
    download_and_extract \
        "gist1m" \
        "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz" \
        "gist" \
        "gist_base.fvecs" \
        "gist_query.fvecs" \
        "gist_groundtruth.ivecs"
fi

echo ""
echo "All requested datasets are ready in $DATA_DIR/"
