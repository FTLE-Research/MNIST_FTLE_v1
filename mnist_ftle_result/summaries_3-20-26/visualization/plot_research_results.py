#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = SCRIPT_DIR.parent
PLOTS_DIR = SCRIPT_DIR / "plots"
DEPTH_SWEEP_SUMMARY = ROOT_DIR / "depth_sweep" / "summary_depth_sweep.csv"
WIDTH_SWEEP_SUMMARY = ROOT_DIR / "width_sweep" / "summary.csv"
SEED_BASELINE_SUMMARY = ROOT_DIR / "seed_stability_baseline" / "summary_seed_baseline.csv"

RAW_PATTERNS = ("*ftle*.csv", "*ftle*.json", "*margin*.csv", "*margin*.json", "*.parquet")
NPZ_PATTERN = "*/artifacts/ftle_margin_data.npz"

FTLE_CANDIDATES = ("ftle", "local_ftle", "lyapunov", "lambda")
MARGIN_CANDIDATES = ("margin", "logit_margin", "signed_margin", "classification_margin")
DEPTH_CANDIDATES = ("depth",)
SEED_CANDIDATES = ("seed",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot depth sweep and FTLE/margin visualizations.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=None,
        help="Optional file or directory containing per-example FTLE/margin data.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=25,
        help="Number of histogram bins and x-axis bins for the binned scatter curve.",
    )
    parser.add_argument(
        "--include-all-widths",
        action="store_true",
        help="Include raw NPZ jobs even if their hyperparameters do not appear in the summary sweep.",
    )
    return parser.parse_args()


def ensure_output_dirs() -> None:
    (SCRIPT_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (SCRIPT_DIR / ".cache").mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_summary_frame() -> pd.DataFrame:
    if not DEPTH_SWEEP_SUMMARY.exists():
        raise FileNotFoundError(f"Missing summary file: {DEPTH_SWEEP_SUMMARY}")
    frame = pd.read_csv(DEPTH_SWEEP_SUMMARY)
    required = {"depth", "seed", "rho_all"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Depth summary is missing required columns: {sorted(missing)}")
    return frame


def load_width_summary_frame() -> pd.DataFrame | None:
    if not WIDTH_SWEEP_SUMMARY.exists():
        return None
    frame = pd.read_csv(WIDTH_SWEEP_SUMMARY)
    required = {"width", "seed", "rho_all"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Width summary is missing required columns: {sorted(missing)}")
    return frame


def save_plot1_rho_vs_depth(summary_df: pd.DataFrame) -> Path:
    stats = summary_df.groupby("depth", as_index=False).agg(
        mean_rho=("rho_all", "mean"),
        std_rho=("rho_all", "std"),
    ).sort_values("depth")
    stats["std_rho"] = stats["std_rho"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        stats["depth"],
        stats["mean_rho"],
        yerr=stats["std_rho"],
        fmt="-o",
        color="#1f5aa6",
        ecolor="#7ea6d8",
        elinewidth=2,
        capsize=4,
        markersize=7,
        linewidth=2.2,
    )
    ax.set_title("Plot 1: Mean rho vs depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Mean rho")
    ax.grid(alpha=0.25, linestyle="--")

    out_path = PLOTS_DIR / "plot1_rho_vs_depth.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_plot4_rho_vs_width(summary_df: pd.DataFrame) -> Path:
    stats = summary_df.groupby("width", as_index=False).agg(
        mean_rho=("rho_all", "mean"),
        std_rho=("rho_all", "std"),
    ).sort_values("width")
    stats["std_rho"] = stats["std_rho"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        stats["width"],
        stats["mean_rho"],
        yerr=stats["std_rho"],
        fmt="-o",
        color="#7b341e",
        ecolor="#f6ad55",
        elinewidth=2,
        capsize=4,
        markersize=7,
        linewidth=2.2,
    )
    ax.set_title("Plot 4: Mean rho vs width")
    ax.set_xlabel("Width")
    ax.set_ylabel("Mean rho")
    ax.grid(alpha=0.25, linestyle="--")

    out_path = PLOTS_DIR / "plot4_rho_vs_width.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def discover_raw_files(search_root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in RAW_PATTERNS:
        files.extend(search_root.rglob(pattern))
    return sorted({path for path in files if path.is_file()})


def discover_npz_files(search_root: Path) -> list[Path]:
    return sorted(path for path in search_root.glob(NPZ_PATTERN) if path.is_file())


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported raw data format: {path}")


def parse_job_metadata(path: Path) -> dict[str, int | float | str]:
    job_name = path.parents[1].name
    match = re.match(
        r"(?P<dataset>[^_]+)_w(?P<width>\d+)_d(?P<depth>\d+)_g(?P<gain>[^_]+)_lr(?P<lr>[^_]+)_bs(?P<batch_size>\d+)_ep(?P<max_epochs>\d+)_seed(?P<seed>\d+)",
        job_name,
    )
    if not match:
        raise ValueError(f"Could not parse job metadata from path: {path}")
    groups = match.groupdict()
    return {
        "dataset": groups["dataset"],
        "width": int(groups["width"]),
        "depth": int(groups["depth"]),
        "gain": float(groups["gain"].replace("p", ".")),
        "lr": float(groups["lr"].replace("p", ".")),
        "batch_size": int(groups["batch_size"]),
        "max_epochs": int(groups["max_epochs"]),
        "seed": int(groups["seed"]),
    }


def npz_to_frame(path: Path) -> pd.DataFrame:
    meta = parse_job_metadata(path)
    data = np.load(path)
    required = {"ftle", "margin"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")

    row_count = len(data["ftle"])
    if len(data["margin"]) != row_count:
        raise ValueError(f"{path} has inconsistent array lengths for ftle and margin")

    frame = pd.DataFrame(
        {
            "depth": np.full(row_count, meta["depth"]),
            "seed": np.full(row_count, meta["seed"]),
            "width": np.full(row_count, meta["width"]),
            "gain": np.full(row_count, meta["gain"]),
            "lr": np.full(row_count, meta["lr"]),
            "batch_size": np.full(row_count, meta["batch_size"]),
            "max_epochs": np.full(row_count, meta["max_epochs"]),
            "ftle": data["ftle"],
            "margin": data["margin"],
        }
    )
    return frame


def rename_first_match(df: pd.DataFrame, candidates: Iterable[str], target: str) -> pd.DataFrame:
    lower_to_original = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return df.rename(columns={lower_to_original[candidate]: target})
    return df


def normalize_raw_frame(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [str(column).strip().lower() for column in renamed.columns]
    renamed = rename_first_match(renamed, FTLE_CANDIDATES, "ftle")
    renamed = rename_first_match(renamed, MARGIN_CANDIDATES, "margin")
    renamed = rename_first_match(renamed, DEPTH_CANDIDATES, "depth")
    renamed = rename_first_match(renamed, SEED_CANDIDATES, "seed")

    required = {"depth", "seed", "ftle", "margin"}
    missing = required - set(renamed.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns after normalization: {sorted(missing)}")

    return renamed[list(required)].copy()


def load_raw_frame(raw_data: Path | None, summary_df: pd.DataFrame, include_all_widths: bool) -> pd.DataFrame | None:
    npz_candidates: list[Path] = []
    candidates: list[Path] = []
    if raw_data is not None:
        if raw_data.is_file():
            if raw_data.suffix == ".npz":
                npz_candidates = [raw_data]
            else:
                candidates = [raw_data]
        elif raw_data.is_dir():
            candidates = discover_raw_files(raw_data)
            npz_candidates = discover_npz_files(raw_data)
        else:
            raise FileNotFoundError(f"Raw data path does not exist: {raw_data}")
    else:
        candidates = discover_raw_files(ROOT_DIR)
        candidates = [path for path in candidates if path.parent != SCRIPT_DIR and "summary" not in path.name.lower()]
        npz_candidates = discover_npz_files(ROOT_DIR / "raw_npz") if (ROOT_DIR / "raw_npz").exists() else []

    frames = []
    for path in npz_candidates:
        try:
            frames.append(npz_to_frame(path))
        except Exception:
            continue

    if not candidates and not frames:
        return None

    for path in candidates:
        try:
            frames.append(normalize_raw_frame(read_table(path), path))
        except Exception:
            continue

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined["depth"] = pd.to_numeric(combined["depth"], errors="coerce")
    combined["seed"] = pd.to_numeric(combined["seed"], errors="coerce")
    combined["ftle"] = pd.to_numeric(combined["ftle"], errors="coerce")
    combined["margin"] = pd.to_numeric(combined["margin"], errors="coerce")
    combined = combined.dropna(subset=["depth", "seed", "ftle", "margin"])

    if not include_all_widths:
        sweep_columns = ["width", "gain", "lr", "batch_size", "max_epochs", "depth", "seed"]
        available_columns = [column for column in sweep_columns if column in combined.columns and column in summary_df.columns]
        if available_columns:
            sweep_keys = summary_df[available_columns].drop_duplicates()
            combined = combined.merge(sweep_keys, on=available_columns, how="inner")

    return combined


def binned_curve(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    work = df.sort_values("margin").copy()
    edges = np.linspace(work["margin"].min(), work["margin"].max(), bins + 1)
    if np.unique(edges).size < 2:
        return pd.DataFrame(columns=["margin_center", "ftle_mean"])

    work["margin_bin"] = pd.cut(work["margin"], bins=edges, include_lowest=True, duplicates="drop")
    curve = work.groupby("margin_bin", observed=True).agg(
        margin_center=("margin", "mean"),
        ftle_mean=("ftle", "mean"),
    )
    return curve.reset_index(drop=True)


def save_plot2_ftle_vs_margin(raw_df: pd.DataFrame, bins: int) -> Path:
    target_depths = [4, 8, 16]
    filtered = raw_df[(raw_df["seed"] == 0) & (raw_df["depth"].isin(target_depths))].copy()
    if filtered.empty:
        raise ValueError("No per-example rows found for seed 0 at depths 4, 8, and 16.")

    fig, axes = plt.subplots(1, len(target_depths), figsize=(16, 4.8), sharey=True)
    colors = {4: "#2b6cb0", 8: "#d97706", 16: "#2f855a"}

    for ax, depth in zip(axes, target_depths):
        depth_df = filtered[filtered["depth"] == depth].copy()
        if depth_df.empty:
            ax.set_visible(False)
            continue

        curve = binned_curve(depth_df, bins=bins)
        ax.scatter(depth_df["margin"], depth_df["ftle"], s=8, alpha=0.18, color=colors[depth])
        if not curve.empty:
            ax.plot(curve["margin_center"], curve["ftle_mean"], color="black", linewidth=2.2)
        ax.set_title(f"Depth {depth}, seed 0")
        ax.set_xlabel("Margin")
        ax.grid(alpha=0.2, linestyle="--")

    axes[0].set_ylabel("FTLE")
    fig.suptitle("Plot 2: FTLE vs margin")

    out_path = PLOTS_DIR / "plot2_ftle_vs_margin_seed0_depths_4_8_16.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_plot3_distributions(raw_df: pd.DataFrame, bins: int) -> Path:
    depths = sorted(int(depth) for depth in raw_df["depth"].dropna().unique())
    if not depths:
        raise ValueError("No depths available in per-example raw data.")

    fig, axes = plt.subplots(len(depths), 2, figsize=(12, 3.2 * len(depths)))
    if len(depths) == 1:
        axes = np.array([axes])

    for row_idx, depth in enumerate(depths):
        depth_df = raw_df[raw_df["depth"] == depth]
        ax_ftle, ax_margin = axes[row_idx]

        ax_ftle.hist(depth_df["ftle"], bins=bins, color="#1f5aa6", alpha=0.8, edgecolor="white")
        ax_ftle.set_title(f"Depth {depth}: FTLE")
        ax_ftle.set_xlabel("FTLE")
        ax_ftle.set_ylabel("Count")
        ax_ftle.grid(alpha=0.15)

        ax_margin.hist(depth_df["margin"], bins=bins, color="#c05621", alpha=0.8, edgecolor="white")
        ax_margin.set_title(f"Depth {depth}: Margin")
        ax_margin.set_xlabel("Margin")
        ax_margin.set_ylabel("Count")
        ax_margin.grid(alpha=0.15)

    fig.suptitle("Plot 3: FTLE and margin distributions by depth")
    out_path = PLOTS_DIR / "plot3_ftle_margin_distributions.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_plot5_ftle_vs_margin_by_width(raw_df: pd.DataFrame, bins: int) -> Path:
    target_widths = [20, 50, 100]
    filtered = raw_df[(raw_df["seed"] == 0) & (raw_df["depth"] == 4) & (raw_df["width"].isin(target_widths))].copy()
    if filtered.empty:
        raise ValueError("No per-example rows found for seed 0 at depth 4 and widths 20, 50, and 100.")

    fig, axes = plt.subplots(1, len(target_widths), figsize=(16, 4.8), sharey=True)
    colors = {20: "#2b6cb0", 50: "#d97706", 100: "#7b341e"}

    for ax, width in zip(axes, target_widths):
        width_df = filtered[filtered["width"] == width].copy()
        if width_df.empty:
            ax.set_visible(False)
            continue

        curve = binned_curve(width_df, bins=bins)
        ax.scatter(width_df["margin"], width_df["ftle"], s=8, alpha=0.18, color=colors[width])
        if not curve.empty:
            ax.plot(curve["margin_center"], curve["ftle_mean"], color="black", linewidth=2.2)
        ax.set_title(f"Width {width}, depth 4, seed 0")
        ax.set_xlabel("Margin")
        ax.grid(alpha=0.2, linestyle="--")

    axes[0].set_ylabel("FTLE")
    fig.suptitle("Plot 5: FTLE vs margin by width")

    out_path = PLOTS_DIR / "plot5_ftle_vs_margin_seed0_widths_20_50_100.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_plot6_distributions_by_width(raw_df: pd.DataFrame, bins: int) -> Path:
    widths = sorted(int(width) for width in raw_df["width"].dropna().unique())
    if not widths:
        raise ValueError("No widths available in per-example raw data.")

    fig, axes = plt.subplots(len(widths), 2, figsize=(12, 3.2 * len(widths)))
    if len(widths) == 1:
        axes = np.array([axes])

    for row_idx, width in enumerate(widths):
        width_df = raw_df[raw_df["width"] == width]
        ax_ftle, ax_margin = axes[row_idx]

        ax_ftle.hist(width_df["ftle"], bins=bins, color="#7b341e", alpha=0.8, edgecolor="white")
        ax_ftle.set_title(f"Width {width}: FTLE")
        ax_ftle.set_xlabel("FTLE")
        ax_ftle.set_ylabel("Count")
        ax_ftle.grid(alpha=0.15)

        ax_margin.hist(width_df["margin"], bins=bins, color="#dd6b20", alpha=0.8, edgecolor="white")
        ax_margin.set_title(f"Width {width}: Margin")
        ax_margin.set_xlabel("Margin")
        ax_margin.set_ylabel("Count")
        ax_margin.grid(alpha=0.15)

    fig.suptitle("Plot 6: FTLE and margin distributions by width")
    out_path = PLOTS_DIR / "plot6_ftle_margin_distributions_by_width.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_status_file(lines: list[str]) -> Path:
    status_path = PLOTS_DIR / "plot_status.txt"
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status_path


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    status_lines: list[str] = []

    summary_df = load_summary_frame()
    plot1_path = save_plot1_rho_vs_depth(summary_df)
    status_lines.append(f"created: {plot1_path.name}")
    status_lines.append(f"summary source: {DEPTH_SWEEP_SUMMARY}")
    if SEED_BASELINE_SUMMARY.exists():
        status_lines.append(f"baseline summary available: {SEED_BASELINE_SUMMARY.name}")
    width_summary_df = load_width_summary_frame()
    if width_summary_df is not None:
        plot4_path = save_plot4_rho_vs_width(width_summary_df)
        status_lines.append(f"created: {plot4_path.name}")
        status_lines.append(f"width summary source: {WIDTH_SWEEP_SUMMARY}")
    else:
        status_lines.append("skipped: plot4_rho_vs_width.png")
        status_lines.append(f"reason: missing width summary file at {WIDTH_SWEEP_SUMMARY}")

    raw_df = load_raw_frame(args.raw_data, summary_df=summary_df, include_all_widths=args.include_all_widths)
    if raw_df is None:
        status_lines.append("skipped: plot2_ftle_vs_margin_seed0_depths_4_8_16.png")
        status_lines.append("reason: no per-example raw data found with columns depth, seed, ftle, margin")
        status_lines.append("skipped: plot3_ftle_margin_distributions.png")
        status_lines.append("reason: no per-example raw data found with columns depth, seed, ftle, margin")
        status_lines.append("skipped: plot5_ftle_vs_margin_seed0_widths_20_50_100.png")
        status_lines.append("reason: no per-example raw data found with columns width, depth, seed, ftle, margin")
        status_lines.append("skipped: plot6_ftle_margin_distributions_by_width.png")
        status_lines.append("reason: no per-example raw data found with columns width, ftle, margin")
        write_status_file(status_lines)
        return

    plot2_path = save_plot2_ftle_vs_margin(raw_df, bins=args.bins)
    plot3_path = save_plot3_distributions(raw_df, bins=args.bins)
    status_lines.append(f"created: {plot2_path.name}")
    status_lines.append(f"created: {plot3_path.name}")
    if width_summary_df is not None and {"width", "depth", "seed", "ftle", "margin"}.issubset(raw_df.columns):
        width_raw_df = raw_df[raw_df["depth"] == 4].copy()
        if not width_raw_df.empty:
            plot5_path = save_plot5_ftle_vs_margin_by_width(width_raw_df, bins=args.bins)
            plot6_path = save_plot6_distributions_by_width(width_raw_df, bins=args.bins)
            status_lines.append(f"created: {plot5_path.name}")
            status_lines.append(f"created: {plot6_path.name}")
        else:
            status_lines.append("skipped: plot5_ftle_vs_margin_seed0_widths_20_50_100.png")
            status_lines.append("reason: no depth-4 per-example raw data available for width sweep")
            status_lines.append("skipped: plot6_ftle_margin_distributions_by_width.png")
            status_lines.append("reason: no depth-4 per-example raw data available for width sweep")
    write_status_file(status_lines)


if __name__ == "__main__":
    main()
