import os
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------
# MNIST version of plot_phase2_phase3_extra.py
#
# Works with:
#   - phase2_grid_state_mnist.npz from phase2_mnist_ftle_vs_margin.py
#   - ra_ka_grid_state_mnist.npz from phase3_ra_ka_grid_mnist.py
#
# Unlike the 2D toy task, MNIST has no global 2D input-space boundary-length metric.
# BL / BE plots will therefore be skipped automatically unless those maps exist and
# contain finite values.
# ---------------------------------------------------------------------

# -------------------- USER CONFIG --------------------
PHASE2_FILE = "phase2_grid_state_mnist.npz"
PHASE3_FILE = "ra_ka_grid_state_mnist.npz"
OUT_DIR = "plots_phase2_phase3_extra_mnist"

USE_LOG10_G_LAMBDA = True
USE_LOG10_POSITIVE_Y_IN_SCATTER = True
EPS = 1e-12

os.makedirs(OUT_DIR, exist_ok=True)


def fmt_tag(x: float) -> str:
    s = f"{float(x):.3g}".replace(".", "p")
    if s.startswith("-"):
        s = "m" + s[1:]
    return s


# -------------------- I/O --------------------
def safe_load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


# -------------------- Stats helpers --------------------
def _rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)

    xs = x[order]
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xx = x[m].astype(np.float64, copy=False)
    yy = y[m].astype(np.float64, copy=False)
    rx = _rankdata_avg_ties(xx)
    ry = _rankdata_avg_ties(yy)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if den == 0.0:
        return float("nan")
    return float((rx * ry).sum() / den)


def nanmean_sem(vals: np.ndarray) -> Tuple[float, float, int]:
    v = vals[np.isfinite(vals)]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=0))
    sem = std / float(np.sqrt(n))
    return mean, float(sem), n


def safe_log10_pos(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    m = np.isfinite(x) & (x > 0)
    out[m] = np.log10(x[m] + eps)
    return out


# -------------------- Flatten grids --------------------
def assert_axes_match(p2: Dict[str, np.ndarray], p3: Dict[str, np.ndarray]) -> None:
    for k in ["widths", "depths", "gains", "base_lrs"]:
        if k not in p2 or k not in p3:
            raise KeyError(f"Missing axis '{k}' in one of the files.")
        if k in ["widths", "depths"]:
            if not np.array_equal(p2[k], p3[k]):
                raise ValueError(f"Axis mismatch for {k}")
        else:
            if not np.allclose(p2[k].astype(np.float64), p3[k].astype(np.float64), rtol=0, atol=0):
                raise ValueError(f"Axis mismatch for {k}")


def flatten_grids(p2: Dict[str, np.ndarray], p3: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    widths = p2["widths"].astype(int)
    depths = p2["depths"].astype(int)
    gains = p2["gains"].astype(np.float64)
    lrs = p2["base_lrs"].astype(np.float64)

    G_lambda = p2["G_lambda_map"].astype(np.float64)
    rho_lam = p2["rho_lambda_map"].astype(np.float64)

    def get_map(name: str) -> np.ndarray:
        if name in p3:
            return p3[name].astype(np.float64)
        return np.full_like(G_lambda, np.nan, dtype=np.float64)

    RA = get_map("RA_map")
    KA = get_map("KA_map")
    FS_cos = get_map("FS_cos_map")
    FS_agree = get_map("FS_agree_map")
    BL = get_map("BL_map")
    NS = get_map("NS_map")
    ND = get_map("ND_map")

    BE = np.full_like(BL, np.nan, dtype=np.float64)
    if "BOUNDARY_GRID" in p3 and "BOUNDARY_BBOX" in p3:
        grid = int(np.array(p3["BOUNDARY_GRID"]).item())
        bbox = np.array(p3["BOUNDARY_BBOX"], dtype=np.float64).reshape(-1)
        if bbox.size == 2 and grid >= 2:
            cell = float((bbox[1] - bbox[0]) / (grid - 1))
            if cell > 0:
                BE = BL / cell

    Gi, Li, Di, Wi = np.meshgrid(
        np.arange(len(gains)),
        np.arange(len(lrs)),
        np.arange(len(depths)),
        np.arange(len(widths)),
        indexing="ij",
    )

    out = {
        "N": widths[Wi].reshape(-1).astype(np.float64),
        "L": depths[Di].reshape(-1).astype(np.float64),
        "g": gains[Gi].reshape(-1).astype(np.float64),
        "lr": lrs[Li].reshape(-1).astype(np.float64),
        "G_lambda": G_lambda.reshape(-1),
        "rho_lambda": rho_lam.reshape(-1),
        "RA": RA.reshape(-1),
        "KA": KA.reshape(-1),
        "FS_cos": FS_cos.reshape(-1),
        "FS_agree": FS_agree.reshape(-1),
        "BL": BL.reshape(-1),
        "BE": BE.reshape(-1),
        "NS": NS.reshape(-1),
        "ND": ND.reshape(-1),
    }

    out["log10_G_lambda"] = safe_log10_pos(out["G_lambda"], eps=EPS)
    out["log10_BL"] = safe_log10_pos(out["BL"], eps=EPS)
    out["log10_NS"] = safe_log10_pos(out["NS"], eps=EPS)
    out["log10_ND"] = safe_log10_pos(out["ND"], eps=EPS)
    return out


# -------------------- Plot helpers --------------------
def plot_metric_vs_axis(axis_vals: np.ndarray, metric_vals: np.ndarray,
                        axis_name: str, metric_name: str, out_path: str):
    m = np.isfinite(axis_vals) & np.isfinite(metric_vals)
    x = axis_vals[m]
    y = metric_vals[m]
    if y.size == 0:
        print(f"[skip] no finite points for {metric_name} vs {axis_name}")
        return

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=12, alpha=0.35)

    uniq = np.unique(x)
    uniq.sort()
    means, sems = [], []
    for u in uniq:
        mu, sem, _ = nanmean_sem(y[x == u])
        means.append(mu)
        sems.append(sem)
    means = np.array(means, dtype=np.float64)
    sems = np.array(sems, dtype=np.float64)

    plt.errorbar(uniq, means, yerr=sems, fmt="-o", capsize=3)
    rho = spearman_rho(x, y)
    plt.title(f"MNIST {metric_name} vs {axis_name}  (n={y.size}, Spearman={rho:.3f})")
    plt.xlabel(axis_name)
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_scatter_pair(x: np.ndarray, y: np.ndarray,
                      x_name: str, y_name: str, out_path: str,
                      title_extra: str = "",
                      xlim: Optional[Tuple[float, float]] = None,
                      ylim: Optional[Tuple[float, float]] = None):
    m = np.isfinite(x) & np.isfinite(y)
    xx = x[m]
    yy = y[m]
    if xx.size == 0:
        print(f"[skip] no finite points for {y_name} vs {x_name} {title_extra}".strip())
        return

    rho = spearman_rho(xx, yy)
    plt.figure(figsize=(6, 4))
    plt.scatter(xx, yy, s=14, alpha=0.45)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    extra = f" {title_extra}".rstrip()
    plt.title(f"MNIST {y_name} vs {x_name}{extra}  (n={xx.size}, Spearman={rho:.3f})")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_scatter_3d_interactive(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                x_name: str, y_name: str, z_name: str,
                                out_html: str,
                                color: Optional[np.ndarray] = None,
                                color_name: str = "",
                                title: Optional[str] = None,
                                auto_open: bool = False):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if color is not None:
        m = m & np.isfinite(color)

    xx = x[m].astype(np.float64, copy=False)
    yy = y[m].astype(np.float64, copy=False)
    zz = z[m].astype(np.float64, copy=False)
    if xx.size == 0:
        print(f"[skip] no finite points for 3D plot {y_name} vs {x_name} vs {z_name}")
        return

    if color is None:
        cc = zz
        ctitle = z_name
    else:
        cc = color[m].astype(np.float64, copy=False)
        ctitle = color_name if color_name else "color"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xx, y=yy, z=zz,
                mode="markers",
                marker=dict(
                    size=3,
                    opacity=0.65,
                    color=cc,
                    showscale=True,
                    colorbar=dict(title=ctitle),
                ),
            )
        ]
    )
    fig.update_layout(
        title=title or f"{y_name} vs {x_name} vs {z_name}",
        scene=dict(xaxis_title=x_name, yaxis_title=y_name, zaxis_title=z_name),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700,
    )
    pio.write_html(fig, file=out_html, include_plotlyjs=True, full_html=True, auto_open=auto_open)
    print(f"[saved interactive] {out_html}")


# -------------------- Main --------------------
def main():
    p2 = safe_load_npz(PHASE2_FILE)
    p3 = safe_load_npz(PHASE3_FILE)
    assert_axes_match(p2, p3)
    flat = flatten_grids(p2, p3)

    if USE_LOG10_G_LAMBDA:
        Gx = flat["log10_G_lambda"]
        Gx_name = "log10(G_lambda)"
    else:
        Gx = flat["G_lambda"]
        Gx_name = "G_lambda"

    # 1) G_lambda vs each axis
    plot_metric_vs_axis(flat["N"], Gx, "N (width)", Gx_name, os.path.join(OUT_DIR, f"{Gx_name}_vs_N.png"))
    plot_metric_vs_axis(flat["L"], Gx, "L (depth)", Gx_name, os.path.join(OUT_DIR, f"{Gx_name}_vs_L.png"))
    plot_metric_vs_axis(flat["g"], Gx, "g (gain)", Gx_name, os.path.join(OUT_DIR, f"{Gx_name}_vs_g.png"))
    plot_metric_vs_axis(flat["lr"], Gx, "lr", Gx_name, os.path.join(OUT_DIR, f"{Gx_name}_vs_lr.png"))

    # 2) rho_lambda vs each axis
    plot_metric_vs_axis(flat["N"], flat["rho_lambda"], "N (width)", "rho_lambda", os.path.join(OUT_DIR, "rho_lambda_vs_N.png"))
    plot_metric_vs_axis(flat["L"], flat["rho_lambda"], "L (depth)", "rho_lambda", os.path.join(OUT_DIR, "rho_lambda_vs_L.png"))
    plot_metric_vs_axis(flat["g"], flat["rho_lambda"], "g (gain)", "rho_lambda", os.path.join(OUT_DIR, "rho_lambda_vs_g.png"))
    plot_metric_vs_axis(flat["lr"], flat["rho_lambda"], "lr", "rho_lambda", os.path.join(OUT_DIR, "rho_lambda_vs_lr.png"))

    # 3) Pairwise scatter at same (N,L,g,lr) cell
    plot_scatter_pair(Gx, flat["RA"], Gx_name, "RA", os.path.join(OUT_DIR, f"RA_vs_{Gx_name}.png"))
    plot_scatter_pair(Gx, flat["KA"], Gx_name, "KA", os.path.join(OUT_DIR, f"KA_vs_{Gx_name}.png"))
    plot_scatter_pair(flat["rho_lambda"], flat["RA"], "rho_lambda", "RA", os.path.join(OUT_DIR, "RA_vs_rho_lambda.png"))
    plot_scatter_pair(flat["rho_lambda"], flat["KA"], "rho_lambda", "KA", os.path.join(OUT_DIR, "KA_vs_rho_lambda.png"))

    plot_scatter_pair(Gx, flat["FS_cos"], Gx_name, "Functional alignment (FS_cos)",
                      os.path.join(OUT_DIR, f"FS_cos_vs_{Gx_name}.png"))
    plot_scatter_pair(flat["rho_lambda"], flat["FS_cos"], "rho_lambda", "Functional alignment (FS_cos)",
                      os.path.join(OUT_DIR, "FS_cos_vs_rho_lambda.png"))

    plot_scatter_pair(Gx, flat["FS_agree"], Gx_name, "Functional agreement (FS_agree)",
                      os.path.join(OUT_DIR, f"FS_agree_vs_{Gx_name}.png"))
    plot_scatter_pair(flat["rho_lambda"], flat["FS_agree"], "rho_lambda", "Functional agreement (FS_agree)",
                      os.path.join(OUT_DIR, "FS_agree_vs_rho_lambda.png"))

    if USE_LOG10_POSITIVE_Y_IN_SCATTER:
        BLy = flat["log10_BL"]
        BLy_name = "log10(Boundary length)"
        NSy = flat["log10_NS"]
        NSy_name = "log10(Noise sensitivity NS)"
        NDy = flat["log10_ND"]
        NDy_name = "log10(Noise acc drop ND)"
    else:
        BLy = flat["BL"]
        BLy_name = "Boundary length"
        NSy = flat["NS"]
        NSy_name = "Noise sensitivity NS"
        NDy = flat["ND"]
        NDy_name = "Noise acc drop ND"

    # For MNIST these will usually skip because BL / BE are NaN.
    plot_scatter_pair(Gx, BLy, Gx_name, BLy_name, os.path.join(OUT_DIR, f"{BLy_name}_vs_{Gx_name}.png"))
    plot_scatter_pair(flat["rho_lambda"], BLy, "rho_lambda", BLy_name, os.path.join(OUT_DIR, f"{BLy_name}_vs_rho_lambda.png"))
    plot_scatter_pair(Gx, flat["BE"], Gx_name, "Boundary edge-count (BE)", os.path.join(OUT_DIR, f"BE_vs_{Gx_name}.png"))
    plot_scatter_pair(flat["rho_lambda"], flat["BE"], "rho_lambda", "Boundary edge-count (BE)", os.path.join(OUT_DIR, "BE_vs_rho_lambda.png"))

    plot_scatter_pair(Gx, NSy, Gx_name, NSy_name, os.path.join(OUT_DIR, f"{NSy_name}_vs_{Gx_name}.png"))
    plot_scatter_pair(flat["rho_lambda"], NSy, "rho_lambda", NSy_name, os.path.join(OUT_DIR, f"{NSy_name}_vs_rho_lambda.png"))
    plot_scatter_pair(Gx, NDy, Gx_name, NDy_name, os.path.join(OUT_DIR, f"{NDy_name}_vs_{Gx_name}.png"))
    plot_scatter_pair(flat["rho_lambda"], NDy, "rho_lambda", NDy_name, os.path.join(OUT_DIR, f"{NDy_name}_vs_rho_lambda.png"))

    # per-g RA vs rho_lambda
    g_vals = flat["g"]
    uniq_g = np.unique(g_vals[np.isfinite(g_vals)])
    uniq_g.sort()
    fixed_xlim = (-1.0, 1.0)
    fixed_ylim = (0.0, 1.0)
    for g0 in uniq_g:
        mg = np.isfinite(flat["rho_lambda"]) & np.isfinite(flat["RA"]) & np.isclose(flat["g"], g0)
        plot_scatter_pair(
            flat["rho_lambda"][mg],
            flat["RA"][mg],
            x_name="rho_lambda",
            y_name="RA",
            out_path=os.path.join(OUT_DIR, f"RA_vs_rho_lambda_g{fmt_tag(g0)}.png"),
            title_extra=f"(g={g0:.3g})",
            xlim=fixed_xlim,
            ylim=fixed_ylim,
        )

    # Optional interactive 3D plot
    plot_scatter_3d_interactive(
        x=flat["rho_lambda"],
        y=flat["RA"],
        z=flat["g"],
        x_name="rho_lambda",
        y_name="RA",
        z_name="gain",
        out_html=os.path.join(OUT_DIR, "RA_vs_rho_lambda_vs_gain.html"),
        color=flat["g"],
        color_name="gain",
        title="MNIST RA vs rho_lambda vs gain",
        auto_open=False,
    )

    print(f"[done] wrote plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
