#!/usr/bin/env python3
"""Generate final-report figures from CUDAsearch benchmark CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MPI = RESULTS / "mpi"
FIGURES = RESULTS / "figures"

COLORS = {
    "old": "#4c78a8",
    "topk": "#f58518",
    "cpu": "#555555",
    "accent": "#54a24b",
    "grid": "#d9d9d9",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def rows_by_rank(path: Path) -> list[dict[str, str]]:
    return sorted(read_csv(path), key=lambda r: as_int(r, "ranks"))


def savefig(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / name
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(out.relative_to(ROOT))


def style_axes(ax) -> None:
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_algorithmic_variant() -> None:
    sift = read_csv(RESULTS / "sift1m_comparison.csv")
    cpu16 = next(r for r in sift if r["kernel"] == "cpu-16")
    cuda_tiled = next(r for r in sift if r["kernel"] == "tiled")
    mpi_tiled_4 = rows_by_rank(MPI / "sift1m_strong_scaling.csv")[-1]
    mpi_topk_4 = rows_by_rank(MPI / "sift1m_tiled_topk_strong_scaling.csv")[-1]

    labels = [
        "CPU-16",
        "CUDA tiled",
        "MPI tiled\n4 ranks",
        "MPI tiled_topk\n4 ranks",
    ]
    latencies = [
        as_float(cpu16, "mean_ms"),
        as_float(cuda_tiled, "mean_ms"),
        as_float(mpi_tiled_4, "mean_ms"),
        as_float(mpi_topk_4, "mean_ms"),
    ]
    colors = [COLORS["cpu"], COLORS["old"], COLORS["old"], COLORS["topk"]]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(labels, latencies, color=colors)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("SIFT1M: GPU top-k turns the bottleneck into a speedup")
    style_axes(ax)

    baseline = latencies[0]
    for bar, latency in zip(bars, latencies):
        speedup = baseline / latency
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            latency + 22,
            f"{latency:.1f} ms\n{speedup:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(latencies) * 1.22)
    savefig("sift1m_algorithmic_variant_latency.png")


def plot_kernel_variants(dataset: str, title_name: str) -> None:
    rows = read_csv(RESULTS / f"{dataset}_comparison.csv")
    order = ["cpu", "cpu-16", "naive", "tiled", "int8", "int8_tiled"]
    labels = {
        "cpu": "CPU",
        "cpu-16": "CPU-16",
        "naive": "CUDA\nnaive",
        "tiled": "CUDA\ntiled",
        "int8": "CUDA\nINT8",
        "int8_tiled": "CUDA\nINT8 tiled",
    }
    rows_by_kernel = {r["kernel"]: r for r in rows}
    kernels = [k for k in order if k in rows_by_kernel]
    latencies = [as_float(rows_by_kernel[k], "mean_ms") for k in kernels]
    recalls = [as_float(rows_by_kernel[k], "recall") for k in kernels]

    fig, ax0 = plt.subplots(figsize=(7.6, 4.2))
    x = list(range(len(kernels)))
    bar_colors = [
        COLORS["cpu"] if k.startswith("cpu") else
        COLORS["topk"] if "int8" in k else
        COLORS["old"]
        for k in kernels
    ]
    bars = ax0.bar(x, latencies, color=bar_colors)
    ax0.set_xticks(x, [labels[k] for k in kernels])
    ax0.set_ylabel("Mean latency (ms)")
    ax0.set_title(f"{title_name}: kernel variants before distributed top-k")
    style_axes(ax0)

    for bar, latency in zip(bars, latencies):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            latency * 1.015,
            f"{latency:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax1 = ax0.twinx()
    ax1.plot(x, recalls, color=COLORS["accent"], marker="o", linewidth=2.0, label="Recall@10")
    ax1.set_ylabel("Recall@10")
    recall_min = max(0.0, min(recalls) - 0.05)
    recall_max = min(1.02, max(recalls) + 0.03)
    ax1.set_ylim(recall_min, recall_max)
    ax1.spines["top"].set_visible(False)
    ax1.legend(frameon=False, loc="upper right")

    savefig(f"{dataset}_kernel_variants_latency_recall.png")


def plot_strong_scaling(dataset: str, title_name: str) -> None:
    old = rows_by_rank(MPI / f"{dataset}_strong_scaling.csv")
    topk = rows_by_rank(MPI / f"{dataset}_tiled_topk_strong_scaling.csv")

    ranks = [as_int(r, "ranks") for r in old]
    old_lat = [as_float(r, "mean_ms") for r in old]
    topk_lat = [as_float(r, "mean_ms") for r in topk]
    old_eff = [(old_lat[0] / y) / p * 100.0 for p, y in zip(ranks, old_lat)]
    topk_eff = [(topk_lat[0] / y) / p * 100.0 for p, y in zip(ranks, topk_lat)]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax0.plot(ranks, old_lat, marker="o", linewidth=2.2, color=COLORS["old"], label="tiled")
    ax0.plot(ranks, topk_lat, marker="o", linewidth=2.2, color=COLORS["topk"], label="tiled_topk")
    ax0.set_xticks(ranks)
    ax0.set_xlabel("MPI ranks / GPUs")
    ax0.set_ylabel("Mean latency (ms)")
    ax0.set_title(f"{title_name} strong scaling")
    ax0.legend(frameon=False)
    style_axes(ax0)

    ax1.plot(ranks, old_eff, marker="o", linewidth=2.2, color=COLORS["old"], label="tiled")
    ax1.plot(ranks, topk_eff, marker="o", linewidth=2.2, color=COLORS["topk"], label="tiled_topk")
    ax1.axhline(100, color="#999999", linewidth=1.0, linestyle="--")
    ax1.set_xticks(ranks)
    ax1.set_xlabel("MPI ranks / GPUs")
    ax1.set_ylabel("Parallel efficiency (%)")
    ax1.set_ylim(0, 115)
    ax1.set_title("Efficiency")
    ax1.legend(frameon=False)
    style_axes(ax1)

    savefig(f"{dataset}_strong_scaling_tiled_vs_topk.png")


def plot_weak_scaling_sift() -> None:
    old = rows_by_rank(MPI / "sift1m_weak_scaling.csv")
    topk = rows_by_rank(MPI / "sift1m_tiled_topk_weak_scaling.csv")

    ranks = [as_int(r, "ranks") for r in old]
    labels = [f"{as_int(r, 'ranks')} rank\nN={as_int(r, 'N')//1000}k" for r in old]
    old_lat = [as_float(r, "mean_ms") for r in old]
    topk_lat = [as_float(r, "mean_ms") for r in topk]

    x = list(range(len(ranks)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar([i - width / 2 for i in x], old_lat, width, label="tiled", color=COLORS["old"])
    ax.bar([i + width / 2 for i in x], topk_lat, width, label="tiled_topk", color=COLORS["topk"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("SIFT1M weak scaling: fixed 250k vectors per rank")
    ax.legend(frameon=False)
    style_axes(ax)
    savefig("sift1m_weak_scaling_tiled_vs_topk.png")


def plot_transfer_volume() -> None:
    before_mb = 100 * 1_000_000 * 4 / 1_000_000
    after_mb = 100 * 10 * (4 + 4) / 1_000_000
    labels = ["old tiled\nB*N scores", "tiled_topk\nB*k pairs"]
    values = [before_mb, after_mb]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(labels, values, color=[COLORS["old"], COLORS["topk"]])
    ax.set_yscale("log")
    ax.set_ylabel("D2H payload per batch (MB, log scale)")
    ax.set_title("GPU top-k removes the score-matrix transfer", pad=16)
    ax.set_ylim(0.004, 2000)
    style_axes(ax)
    for bar, value in zip(bars, values):
        label = f"{value:.3g} MB" if value < 1 else f"{value:.0f} MB"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.25,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )
    savefig("topk_d2h_payload_reduction.png")


def main() -> None:
    plot_kernel_variants("sift1m", "SIFT1M")
    plot_kernel_variants("gist1m", "GIST1M")
    plot_algorithmic_variant()
    plot_strong_scaling("sift1m", "SIFT1M")
    plot_weak_scaling_sift()
    plot_strong_scaling("gist1m", "GIST1M")
    plot_transfer_volume()


if __name__ == "__main__":
    main()
