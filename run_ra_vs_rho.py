from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models import make_model
from utils import DEVICE, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RA and plot RA vs rho_lambda for trained MNIST sweeps.")
    parser.add_argument(
        "--summary-root",
        type=Path,
        required=True,
        help="Directory containing summary subdirectories such as depth_sweep/ and width_sweep/.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Root directory containing runs/jobs/mnist/<job_id>/checkpoints/best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/summaries/ra_vs_rho"),
        help="Directory for merged tables and plots.",
    )
    parser.add_argument(
        "--split",
        choices=("test", "train"),
        default="test",
        help="Dataset split used to build Z_init and Z_final.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=2000,
        help="Use the first N examples from the chosen split. Use -1 for the full split.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best.pt",
        help="Checkpoint filename under each job's checkpoints/ directory.",
    )
    return parser.parse_args()


def _summary_file(summary_root: Path, experiment_name: str) -> Path:
    candidates = [
        summary_root / experiment_name / "summary.csv",
        summary_root / experiment_name / f"summary_{experiment_name}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find summary CSV for experiment={experiment_name!r} under {summary_root}")


def load_summary(summary_root: Path, experiment_name: str) -> pd.DataFrame:
    path = _summary_file(summary_root, experiment_name)
    frame = pd.read_csv(path)
    required = {"job_id", "width", "depth", "gain", "lr", "batch_size", "max_epochs", "seed", "rho_all"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Summary {path} is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["experiment_name"] = experiment_name
    frame["summary_file"] = str(path)
    return frame


def load_representation_inputs(split: str, subset: int) -> torch.Tensor:
    from data import load_mnist_tensors

    train_ds, test_ds = load_mnist_tensors(normalize=False)
    dataset = test_ds if split == "test" else train_ds
    x_all, _ = dataset.tensors
    if subset >= 0:
        x_all = x_all[:subset]
    return x_all


@torch.no_grad()
def hidden_matrix(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(DEVICE, non_blocking=True)
        parts.append(model.hidden_map(xb).detach().cpu())
    return torch.cat(parts, dim=0)


def representation_alignment(z_init: torch.Tensor, z_final: torch.Tensor) -> float:
    zi = z_init.reshape(-1).double()
    zf = z_final.reshape(-1).double()
    denom = torch.linalg.norm(zi) * torch.linalg.norm(zf)
    if denom.item() == 0.0:
        return float("nan")
    return float(torch.dot(zi, zf).item() / denom.item())


def build_init_model(width: int, depth: int, gain: float, seed: int) -> torch.nn.Module:
    set_seed(seed)
    model = make_model(width=width, depth=depth, gain=gain).to(DEVICE)
    model.eval()
    return model


def build_final_model(width: int, depth: int, gain: float, checkpoint_path: Path) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location=DEVICE)
    model = make_model(width=width, depth=depth, gain=gain).to(DEVICE)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def compute_ra_for_row(row: pd.Series, checkpoint_root: Path, checkpoint_name: str, x_repr: torch.Tensor) -> dict[str, object]:
    job_id = str(row["job_id"])
    checkpoint_path = checkpoint_root / job_id / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {job_id}: {checkpoint_path}")

    width = int(row["width"])
    depth = int(row["depth"])
    gain = float(row["gain"])
    seed = int(row["seed"])

    init_model = build_init_model(width=width, depth=depth, gain=gain, seed=seed)
    final_model = build_final_model(width=width, depth=depth, gain=gain, checkpoint_path=checkpoint_path)

    z_init = hidden_matrix(init_model, x_repr)
    z_final = hidden_matrix(final_model, x_repr)
    ra = representation_alignment(z_init, z_final)

    return {
        "experiment_name": str(row["experiment_name"]),
        "job_id": job_id,
        "width": width,
        "depth": depth,
        "gain": gain,
        "lr": float(row["lr"]),
        "batch_size": int(row["batch_size"]),
        "max_epochs": int(row["max_epochs"]),
        "seed": seed,
        "ra": ra,
        "rho_lambda": float(row["rho_all"]),
        "checkpoint_path": str(checkpoint_path),
    }


def save_scatter(frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {
        "depth_sweep": "#1f5aa6",
        "width_sweep": "#a63d1f",
    }
    for experiment_name, group in frame.groupby("experiment_name"):
        ax.scatter(
            group["ra"],
            group["rho_lambda"],
            s=40,
            alpha=0.85,
            label=experiment_name,
            color=colors.get(experiment_name, None),
        )

    ax.set_xlabel("RA")
    ax.set_ylabel(r"$\rho_\lambda$")
    ax.set_title("RA vs FTLE Predictability")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_group_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    depth_df = frame[frame["experiment_name"] == "depth_sweep"]
    if not depth_df.empty:
        stats = depth_df.groupby("depth", as_index=False).agg(
            mean_ra=("ra", "mean"),
            std_ra=("ra", "std"),
            mean_rho=("rho_lambda", "mean"),
            std_rho=("rho_lambda", "std"),
        ).sort_values("depth")
        stats = stats.fillna(0.0)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(stats["depth"], stats["mean_ra"], yerr=stats["std_ra"], fmt="-o", color="#1f5aa6", capsize=4)
        ax.set_xlabel("Depth")
        ax.set_ylabel("Mean RA")
        ax.set_title("Mean RA vs Depth")
        ax.grid(alpha=0.25, linestyle="--")
        fig.tight_layout()
        fig.savefig(output_dir / "mean_ra_vs_depth.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(stats["depth"], stats["mean_rho"], yerr=stats["std_rho"], fmt="-o", color="#1f5aa6", capsize=4)
        ax.set_xlabel("Depth")
        ax.set_ylabel(r"Mean $\rho_\lambda$")
        ax.set_title("Mean FTLE Predictability vs Depth")
        ax.grid(alpha=0.25, linestyle="--")
        fig.tight_layout()
        fig.savefig(output_dir / "mean_rho_vs_depth_from_ra_run.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    width_df = frame[frame["experiment_name"] == "width_sweep"]
    if not width_df.empty:
        stats = width_df.groupby("width", as_index=False).agg(
            mean_ra=("ra", "mean"),
            std_ra=("ra", "std"),
            mean_rho=("rho_lambda", "mean"),
            std_rho=("rho_lambda", "std"),
        ).sort_values("width")
        stats = stats.fillna(0.0)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(stats["width"], stats["mean_ra"], yerr=stats["std_ra"], fmt="-o", color="#a63d1f", capsize=4)
        ax.set_xlabel("Width")
        ax.set_ylabel("Mean RA")
        ax.set_title("Mean RA vs Width")
        ax.grid(alpha=0.25, linestyle="--")
        fig.tight_layout()
        fig.savefig(output_dir / "mean_ra_vs_width.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(stats["width"], stats["mean_rho"], yerr=stats["std_rho"], fmt="-o", color="#a63d1f", capsize=4)
        ax.set_xlabel("Width")
        ax.set_ylabel(r"Mean $\rho_\lambda$")
        ax.set_title("Mean FTLE Predictability vs Width")
        ax.grid(alpha=0.25, linestyle="--")
        fig.tight_layout()
        fig.savefig(output_dir / "mean_rho_vs_width_from_ra_run.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_frames = [
        load_summary(args.summary_root, "depth_sweep"),
        load_summary(args.summary_root, "width_sweep"),
    ]
    merged_summary = pd.concat(summary_frames, ignore_index=True)

    x_repr = load_representation_inputs(split=args.split, subset=args.subset)
    rows = []
    for row in merged_summary.itertuples(index=False):
        rows.append(
            compute_ra_for_row(
                row=pd.Series(row._asdict()),
                checkpoint_root=args.checkpoint_root,
                checkpoint_name=args.checkpoint_name,
                x_repr=x_repr,
            )
        )

    result_df = pd.DataFrame(rows).sort_values(["experiment_name", "depth", "width", "seed"]).reset_index(drop=True)
    csv_path = args.output_dir / "ra_vs_rho.csv"
    json_path = args.output_dir / "ra_vs_rho.json"
    scatter_path = args.output_dir / "ra_vs_rho_scatter.png"

    result_df.to_csv(csv_path, index=False)
    json_path.write_text(result_df.to_json(orient="records", indent=2), encoding="utf-8")
    save_scatter(result_df, scatter_path)
    save_group_plots(result_df, args.output_dir)

    summary_payload = {
        "summary_root": str(args.summary_root),
        "checkpoint_root": str(args.checkpoint_root),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "subset": args.subset,
        "checkpoint_name": args.checkpoint_name,
        "jobs_processed": int(len(result_df)),
        "mean_ra": float(result_df["ra"].mean()),
        "mean_rho_lambda": float(result_df["rho_lambda"].mean()),
        "csv": str(csv_path),
        "json": str(json_path),
        "scatter_plot": str(scatter_path),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(summary_payload)


if __name__ == "__main__":
    main()
