"""
MNIST version of phase2_ftle_vs_margin.py.

Important differences from the 2D circle version:
1) MNIST input is 784-D, so there is NO 2D FTLE grid / bbox / ridge geometry here.
   We compute per-sample FTLE directly on an evaluation subset using batched power iteration
   on the Jacobian of the last hidden representation wrt input.
2) This is multiclass (10-way) classification, so the model outputs logits and training uses CE.
3) Adversarial margin is the minimum L_inf perturbation (via PGD + bisection) that changes the
   predicted class away from the true label.

This script preserves the same broad outputs:
- per-seed caches with ftle_vals, margins, G_lambda, G_J, rho_lambda_margin, rho_J_margin
- aggregated 4D maps over (gain, lr, depth, width)

If you want boundary/ridge geometry on MNIST later, the right analogue is to study 2D input slices
(or latent-space slices), not a full 784-D grid.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import contextlib
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from ra_ka_mnist_accstop import (
    FC,
    load_or_make_mnist_data,
    verify_or_train_checkpoint,
    dataset_to_loader,
    DEVICE,
    TRAIN_ACC_TARGET,
    MAX_EPOCHS,
    BATCH_SIZE_TRAIN,
    fmt_float,
)

# -------------------- GPU knobs --------------------
device = DEVICE
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

# -------------------- USER CONFIG --------------------
# Start small for MNIST; expand after confirming training works.
WIDTHS   = [32, 64, 128, 256, 512, 1024, 2048]
DEPTHS   = [2, 4, 6, 8]
GAINS    = [0.8, 0.9, 1.0, 1.1, 1.2]
BASE_LRS = [5.0, 10.0, 20.0]
SEEDS    = [0, 1, 2]

# Attack / evaluation
EPS_HI          = 0.30   # L_inf in the centered [−0.5, 0.5] pixel scale
PGD_STEPS       = 20
BISECTION_ITERS = 10

# Evaluation subset (full MNIST test set is expensive for FTLE + PGD)
EVAL_SUBSET      = 1024
EVAL_SUBSET_SEED = 1234

# FTLE via batched power iteration
FTLE_POWER_ITERS = 8
FTLE_BATCH       = 128

# Resume / caching
CACHE_DIR   = "phase2_cache_mnist"
GRID_STATE  = "phase2_grid_state_mnist.npz"
PLOT_DIR    = "plots_mnist"

# Data caching
DATA_SEED       = 0
TRAIN_LIMIT     = None   # e.g. 20000 for faster experiments
TEST_LIMIT      = None   # None means use all official test points before subselecting eval subset
DATA_CACHE_FILE = f"mnist_centered_seed{DATA_SEED}_train{TRAIN_LIMIT}_test{TEST_LIMIT}.npz"

# Save partial seed progress every N points
SAVE_EVERY_POINTS = 200

# Control behavior
DO_COMPUTE = True
DO_PLOT    = True

# -------------------- SPEED KNOBS --------------------
MARGIN_BATCH = 256
USE_AMP      = True
AMP_DTYPE    = torch.bfloat16
USE_COMPILE  = False
LOSS_FP32    = True
CUDA_EMPTY_CACHE_EACH_SEED = False

# -------------------- VERSIONING --------------------
CACHE_VERSION = 1
GRID_VERSION  = 1

# -------------------- NUMERICAL SAFETY --------------------
LOG_EXP_CLIP = 700.0

# -------------------- torch.func transforms --------------------
try:
    from torch.func import jvp, vjp
except Exception as e:
    raise RuntimeError("This script needs torch.func.jvp and torch.func.vjp (PyTorch >= 2.0).") from e


def autocast_ctx():
    if USE_AMP and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return contextlib.nullcontext()


def atomic_save_npz(path: str, **arrays) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def safe_load_npz(path: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return None


def sanitize_lambda(arr: np.ndarray) -> np.ndarray:
    lam = np.asarray(arr, dtype=np.float64)
    lam = np.array(lam, copy=True)
    lam[~np.isfinite(lam)] = np.nan
    return lam


def spearman_rho_only(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = rankdata(x[m], method="average")
    ry = rankdata(y[m], method="average")
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


# -------------------- Data --------------------
def load_mnist_eval_subset() -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    (xt, yt), (xe, ye) = load_or_make_mnist_data(
        DATA_CACHE_FILE,
        seed=DATA_SEED,
        train_limit=TRAIN_LIMIT,
        test_limit=TEST_LIMIT,
        digits=None,
        center=True,
    )

    g = torch.Generator(device="cpu").manual_seed(EVAL_SUBSET_SEED)
    n_eval = min(EVAL_SUBSET, xe.shape[0])
    idx = torch.randperm(xe.shape[0], generator=g)[:n_eval]
    xe_eval = xe[idx]
    ye_eval = ye[idx]
    return (xt, yt), (xe_eval, ye_eval)


# -------------------- Checkpoint loading --------------------
def load_or_train_net(N: int, L: int, gain: float, base_lr: float, seed: int, train_loader) -> Optional[FC]:
    net = verify_or_train_checkpoint(
        N, L, gain, base_lr, seed,
        train_loader=train_loader,
        acc_target=TRAIN_ACC_TARGET,
        max_epochs=MAX_EPOCHS,
        fail_policy="none",
    )
    if net is None:
        print(f"[skip-model] N={N} L={L} g={gain} lr={base_lr} seed={seed} failed to reach acc_target={TRAIN_ACC_TARGET:.3f}")
        return None
    net.eval()
    return net


# -------------------- FTLE per-sample (no grid in MNIST) --------------------
def _row_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1, keepdim=True) + eps


def sigma_max_hidden_batch(net: FC, X: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """
    Approximate sigma_max(d h_L / d x) per sample in a batch using batched power iteration.
    X: [B,784]
    Returns: sigma [B]
    """
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    def hidden_fn(z: torch.Tensor) -> torch.Tensor:
        return net(z, hid=True)

    v = torch.randn_like(X)
    v = v / _row_norm(v)

    for _ in range(iters):
        _, Jv = jvp(hidden_fn, (X,), (v,))
        u = Jv / _row_norm(Jv)

        _, vjp_fn = vjp(hidden_fn, X)
        JT_u = vjp_fn(u)[0]
        v = JT_u / _row_norm(JT_u)

    _, Jv = jvp(hidden_fn, (X,), (v,))
    sigma = _row_norm(Jv).squeeze(1)
    return sigma.float()


def compute_ftle_vals(net: FC, X: torch.Tensor, depth: int, batch_size: int, iters: int) -> np.ndarray:
    vals = []
    for s in range(0, X.shape[0], batch_size):
        xb = X[s:s + batch_size]
        sigma = sigma_max_hidden_batch(net, xb, iters=iters)
        lam = (1.0 / depth) * torch.log(sigma + 1e-12)
        vals.append(lam.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(vals, axis=0)


# -------------------- Batched PGD + margin (multiclass) --------------------
def pgd_batch_multiclass(net: FC, X: torch.Tensor, y: torch.Tensor, eps: torch.Tensor, k: int) -> torch.Tensor:
    """
    Untargeted L_inf PGD for multiclass logits.
    X: [B,784], y: [B], eps: [B]
    Inputs are assumed to live in [-0.5, 0.5].
    """
    B = X.shape[0]
    eps2 = eps.view(B, 1)
    step = eps2 / 10.0
    delta = torch.zeros_like(X)

    for _ in range(k):
        delta.requires_grad_(True)
        with autocast_ctx():
            logits = net(X + delta)
            if LOSS_FP32:
                loss = torch.nn.functional.cross_entropy(logits.float(), y, reduction="sum")
            else:
                loss = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
        grad = torch.autograd.grad(loss, delta, create_graph=False, retain_graph=False)[0]
        with torch.no_grad():
            delta.add_(step * grad.sign())
            delta.clamp_(-eps2, eps2)
            # keep in valid centered pixel range
            delta.copy_((X + delta).clamp_(-0.5, 0.5) - X)
        delta = delta.detach()

    return (X + delta).detach()


def margin_batch(net: FC, X: torch.Tensor, y: torch.Tensor,
                 eps_hi: float, bisection_iters: int, pgd_steps: int) -> torch.Tensor:
    B = X.shape[0]
    lo = torch.zeros((B,), device=X.device, dtype=X.dtype)
    hi = torch.full((B,), eps_hi, device=X.device, dtype=X.dtype)

    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        adv = pgd_batch_multiclass(net, X, y, eps=mid, k=pgd_steps)
        with torch.inference_mode():
            with autocast_ctx():
                pred = torch.argmax(net(adv), dim=1)
            success = pred.ne(y)
        hi = torch.where(success, mid, hi)
        lo = torch.where(success, lo, mid)
    return hi


# -------------------- Per-seed caching --------------------
def seed_cache_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    gstr = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    return os.path.join(
        CACHE_DIR,
        f"seedstats_mnist_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}_eval{EVAL_SUBSET}_dseed{DATA_SEED}.npz",
    )


def cache_version_of(d: Optional[Dict[str, np.ndarray]]) -> int:
    if d is None or "cache_version" not in d:
        return 0
    return int(np.array(d["cache_version"]).item())


def is_finished_seed_cache(d: Optional[Dict[str, np.ndarray]], n_test: int) -> bool:
    if d is None:
        return False
    if not bool(np.array(d.get("finished", False)).item()):
        return False
    if cache_version_of(d) != CACHE_VERSION:
        return False
    if "margins" not in d or d["margins"].shape[0] != n_test:
        return False
    if "ftle_vals" not in d or d["ftle_vals"].shape[0] != n_test:
        return False
    return True


def compute_or_resume_seed_stats(
    N: int, L: int, gain: float, base_lr: float, seed: int,
    train_loader,
    X_test: torch.Tensor, y_test: torch.Tensor,
) -> Dict[str, np.ndarray]:
    path = seed_cache_path(N, L, gain, base_lr, seed)
    n_test = X_test.shape[0]

    net = load_or_train_net(N, L, gain, base_lr, seed, train_loader=train_loader)
    if net is None:
        margins = np.full(n_test, np.nan, dtype=np.float32)
        ftle_vals = np.full(n_test, np.nan, dtype=np.float32)
        out = dict(
            cache_version=np.array(CACHE_VERSION, dtype=np.int32),
            finished=np.array(True),
            train_ok=np.array(False),
            N=np.array(N), L=np.array(L),
            gain=np.array(gain, dtype=np.float32),
            base_lr=np.array(base_lr, dtype=np.float32),
            seed=np.array(seed, dtype=np.int32),
            n_test=np.array(n_test, dtype=np.int32),
            margins=margins,
            ftle_vals=ftle_vals,
            G_lambda=np.array(np.nan, dtype=np.float64),
            G_J=np.array(np.nan, dtype=np.float64),
            rho_lambda_margin=np.array(np.nan, dtype=np.float64),
            rho_J_margin=np.array(np.nan, dtype=np.float64),
        )
        atomic_save_npz(path, **out)
        return out

    cached = safe_load_npz(path) if os.path.exists(path) else None
    if is_finished_seed_cache(cached, n_test):
        return cached

    # Reuse margins if present
    if cached is not None and "margins" in cached and cached["margins"].shape[0] == n_test:
        margins = cached["margins"].astype(np.float32, copy=True)
    else:
        margins = np.full(n_test, np.nan, dtype=np.float32)

    # Recompute FTLE values directly per sample (no grid)
    ftle_vals = compute_ftle_vals(net, X_test, depth=L, batch_size=FTLE_BATCH, iters=FTLE_POWER_ITERS)

    todo = np.where(~np.isfinite(margins))[0]
    if todo.size:
        net.eval()
        if USE_COMPILE and hasattr(torch, "compile") and device.type == "cuda":
            net = torch.compile(net, mode="reduce-overhead")
        for p in net.parameters():
            p.requires_grad_(False)

        t0 = time.time()
        done = 0
        last_save = 0
        for start in range(0, todo.size, MARGIN_BATCH):
            idx_np = todo[start:start + MARGIN_BATCH]
            idx_t = torch.as_tensor(idx_np, device=X_test.device, dtype=torch.long)

            eps_star = margin_batch(
                net,
                X_test[idx_t],
                y_test[idx_t],
                eps_hi=EPS_HI,
                bisection_iters=BISECTION_ITERS,
                pgd_steps=PGD_STEPS,
            )
            margins[idx_np] = eps_star.float().cpu().numpy()
            done += idx_np.size

            if (done - last_save) >= SAVE_EVERY_POINTS or (done == todo.size):
                last_save = done
                atomic_save_npz(
                    path,
                    cache_version=np.array(CACHE_VERSION, np.int32),
                    finished=np.array(False),
                    N=np.array(N, np.int32), L=np.array(L, np.int32),
                    gain=np.array(gain, np.float32),
                    base_lr=np.array(base_lr, np.float32),
                    seed=np.array(seed, np.int32),
                    n_test=np.array(n_test, np.int32),
                    margins=margins,
                    ftle_vals=ftle_vals,
                )
                dt = (time.time() - t0) / 60.0
                print(f"[save-partial] {os.path.basename(path)}  done={int(np.isfinite(margins).sum())}/{n_test}  dt={dt:.1f} min")

        del net
        if CUDA_EMPTY_CACHE_EACH_SEED and device.type == "cuda":
            torch.cuda.empty_cache()

    lam = sanitize_lambda(ftle_vals)
    G_lambda = float(np.nanvar(lam))
    jac_norms = np.exp(np.clip(L * lam, -LOG_EXP_CLIP, LOG_EXP_CLIP))
    G_J = float(np.nanvar(jac_norms))

    rho_lambda_seed = spearman_rho_only(lam, margins.astype(np.float64))
    rho_J_seed = spearman_rho_only(jac_norms, margins.astype(np.float64))

    out = dict(
        cache_version=np.array(CACHE_VERSION, np.int32),
        finished=np.array(True),
        train_ok=np.array(True),
        N=np.array(N, np.int32), L=np.array(L, np.int32),
        gain=np.array(gain, np.float32),
        base_lr=np.array(base_lr, np.float32),
        seed=np.array(seed, np.int32),
        n_test=np.array(n_test, np.int32),
        margins=margins.astype(np.float32),
        ftle_vals=ftle_vals.astype(np.float32),
        G_lambda=np.array(G_lambda, np.float64),
        G_J=np.array(G_J, np.float64),
        rho_lambda_margin=np.array(rho_lambda_seed, np.float64),
        rho_J_margin=np.array(rho_J_seed, np.float64),
    )
    atomic_save_npz(path, **out)
    return out


# -------------------- Aggregation over seeds --------------------
def aggregate_config_pooled(seed_dicts: List[Dict[str, np.ndarray]], L: int) -> Dict[str, float]:
    good = []
    for d in seed_dicts:
        ok = bool(np.array(d.get("train_ok", True)).item())
        if ok:
            good.append(d)

    if len(good) == 0:
        return dict(
            G_lambda_mean=float("nan"),
            G_J_mean=float("nan"),
            rho_lambda_mean=float("nan"),
            rho_J_mean=float("nan"),
            sat_frac=float("nan"),
            rho_lambda_unsat=float("nan"),
            n_good=0,
        )

    G_lambda_mean = float(np.nanmean([float(d["G_lambda"]) for d in good]))
    G_J_mean = float(np.nanmean([float(d["G_J"]) for d in good]))

    lam_all = np.concatenate([sanitize_lambda(d["ftle_vals"]) for d in good], axis=0)
    m_all = np.concatenate([d["margins"].astype(np.float64, copy=False) for d in good], axis=0)

    rho_lambda = spearman_rho_only(lam_all, m_all)
    jac_all = np.exp(np.clip(L * lam_all, -LOG_EXP_CLIP, LOG_EXP_CLIP))
    rho_J = spearman_rho_only(jac_all, m_all)

    finite_m = np.isfinite(m_all)
    if finite_m.sum() == 0:
        sat_frac = float("nan")
        rho_lambda_unsat = float("nan")
    else:
        sat_mask = finite_m & (m_all >= (EPS_HI - 1e-6))
        sat_frac = float(sat_mask.sum() / finite_m.sum())
        unsat_mask = finite_m & (m_all < (EPS_HI - 1e-6))
        rho_lambda_unsat = spearman_rho_only(lam_all[unsat_mask], m_all[unsat_mask]) if unsat_mask.sum() >= 3 else float("nan")

    return dict(
        G_lambda_mean=G_lambda_mean,
        G_J_mean=G_J_mean,
        rho_lambda_mean=rho_lambda,
        rho_J_mean=rho_J,
        sat_frac=sat_frac,
        rho_lambda_unsat=rho_lambda_unsat,
        n_good=len(good),
    )


# -------------------- Grid state --------------------
def save_grid_state(path: str, widths, depths, gains, base_lrs, seeds,
                    G_lambda_map, G_J_map, rho_lambda_map, rho_J_map, done_map):
    atomic_save_npz(
        path,
        grid_version=np.array(GRID_VERSION, np.int32),
        data_seed=np.array(DATA_SEED, np.int32),
        widths=np.array(widths, np.int32),
        depths=np.array(depths, np.int32),
        gains=np.array(gains, np.float32),
        base_lrs=np.array(base_lrs, np.float32),
        seeds=np.array(seeds, np.int32),
        G_lambda_map=G_lambda_map.astype(np.float64),
        G_J_map=G_J_map.astype(np.float64),
        rho_lambda_map=rho_lambda_map.astype(np.float64),
        rho_J_map=rho_J_map.astype(np.float64),
        done_map=done_map.astype(np.bool_),
    )


def try_load_grid_state(path: str, widths, depths, gains, base_lrs, seeds):
    if not os.path.exists(path):
        return None
    d = safe_load_npz(path)
    if d is None:
        return None
    if int(np.array(d.get("grid_version", 0)).item()) != GRID_VERSION:
        return None
    if int(np.array(d.get("data_seed", -1)).item()) != DATA_SEED:
        return None
    if (not np.array_equal(np.array(widths), d.get("widths")) or
        not np.array_equal(np.array(depths), d.get("depths")) or
        not np.allclose(np.array(gains, np.float32), d.get("gains").astype(np.float32)) or
        not np.allclose(np.array(base_lrs, np.float32), d.get("base_lrs").astype(np.float32)) or
        not np.array_equal(np.array(seeds), d.get("seeds"))):
        return None
    return d


def run_grid_resume(widths, depths, gains, base_lrs, seeds, train_loader, X_test, y_test):
    shape = (len(gains), len(base_lrs), len(depths), len(widths))
    loaded = try_load_grid_state(GRID_STATE, widths, depths, gains, base_lrs, seeds)
    if loaded is not None:
        G_lambda_map = loaded["G_lambda_map"]
        G_J_map = loaded["G_J_map"]
        rho_lambda_map = loaded["rho_lambda_map"]
        rho_J_map = loaded["rho_J_map"]
        done_map = loaded["done_map"].astype(bool)
        print(f"[grid-state] loaded {done_map.sum()}/{done_map.size} cells")
    else:
        G_lambda_map = np.full(shape, np.nan, np.float64)
        G_J_map = np.full(shape, np.nan, np.float64)
        rho_lambda_map = np.full(shape, np.nan, np.float64)
        rho_J_map = np.full(shape, np.nan, np.float64)
        done_map = np.zeros(shape, bool)

    total = done_map.size
    for gi, gain in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    if done_map[gi, li, di, wi]:
                        continue
                    print(f"\n[cell] N={N} L={L} g={gain} lr={lr}")
                    seed_stats = []
                    for sd in seeds:
                        seed_stats.append(
                            compute_or_resume_seed_stats(
                                N, L, gain, lr, sd,
                                train_loader=train_loader,
                                X_test=X_test,
                                y_test=y_test,
                            )
                        )

                    agg = aggregate_config_pooled(seed_stats, L=L)
                    print(f"[cell-done] seeds_used={agg.get('n_good', 'NA')}/{len(seeds)} ...")

                    G_lambda_map[gi, li, di, wi] = agg["G_lambda_mean"]
                    G_J_map[gi, li, di, wi] = agg["G_J_mean"]
                    rho_lambda_map[gi, li, di, wi] = agg["rho_lambda_mean"]
                    rho_J_map[gi, li, di, wi] = agg["rho_J_mean"]
                    done_map[gi, li, di, wi] = True

                    done = int(done_map.sum())
                    print(f"[cell-done] ({done}/{total})  Gλ={agg['G_lambda_mean']:.3e}  ρλ={agg['rho_lambda_mean']:.3f}  ρλ(unsat)={agg['rho_lambda_unsat']:.3f}  sat={agg['sat_frac']:.3f}")

                    save_grid_state(
                        GRID_STATE,
                        widths, depths, gains, base_lrs, seeds,
                        G_lambda_map, G_J_map, rho_lambda_map, rho_J_map, done_map,
                    )

    save_grid_state(
        GRID_STATE,
        widths, depths, gains, base_lrs, seeds,
        G_lambda_map, G_J_map, rho_lambda_map, rho_J_map, done_map,
    )
    return dict(
        widths=widths, depths=depths, gains=gains, base_lrs=base_lrs, seeds=seeds,
        G_lambda_map=G_lambda_map, G_J_map=G_J_map,
        rho_lambda_map=rho_lambda_map, rho_J_map=rho_J_map,
        done_map=done_map,
    )


# -------------------- Plotting --------------------
def plot_heatmap(mat2d: np.ndarray, widths: List[int], depths: List[int],
                 title: str, out_path: str, vmin=None, vmax=None, log10: bool = False):
    plt.figure(figsize=(6, 4))
    M = mat2d.copy()
    if log10:
        M = np.log10(M + 1e-12)
    M = np.ma.masked_invalid(M)
    im = plt.imshow(M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("Width N")
    plt.ylabel("Depth L")
    plt.xticks(range(len(widths)), widths)
    plt.yticks(range(len(depths)), depths)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_all_slices(grid: Dict):
    widths = grid["widths"]
    depths = grid["depths"]
    gains = grid["gains"]
    base_lrs = grid["base_lrs"]

    G_lambda_map = grid["G_lambda_map"]
    rho_lambda_map = grid["rho_lambda_map"]
    G_J_map = grid["G_J_map"]
    rho_J_map = grid["rho_J_map"]

    for gi, g in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            gstr = fmt_float(float(g))
            lrstr = fmt_float(float(lr))
            plot_heatmap(G_lambda_map[gi, li], widths, depths,
                         title=f"MNIST log10 Var[λ]  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_log10_Glambda_g{gstr}_lr{lrstr}.png"),
                         log10=True)
            plot_heatmap(rho_lambda_map[gi, li], widths, depths,
                         title=f"MNIST rho(λ, margin) [pooled]  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_rho_lambda_g{gstr}_lr{lrstr}.png"),
                         vmin=-1, vmax=1)
            plot_heatmap(G_J_map[gi, li], widths, depths,
                         title=f"MNIST log10 Var[||J||]  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_log10_GJ_g{gstr}_lr{lrstr}.png"),
                         log10=True)
            plot_heatmap(rho_J_map[gi, li], widths, depths,
                         title=f"MNIST rho(||J||, margin) [pooled]  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_rho_J_g{gstr}_lr{lrstr}.png"),
                         vmin=-1, vmax=1)


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("[device]", device)
    if device.type == "cuda":
        print("[gpu]", torch.cuda.get_device_name(0))

    (xt, yt), (xe, ye) = load_mnist_eval_subset()
    train_loader = dataset_to_loader((xt, yt), BATCH_SIZE_TRAIN, shuffle=True, device=device)
    X_test = xe.to(device)
    y_test = ye.to(device)

    if DO_COMPUTE:
        grid = run_grid_resume(
            WIDTHS, DEPTHS, GAINS, BASE_LRS, SEEDS,
            train_loader=train_loader,
            X_test=X_test, y_test=y_test,
        )
    else:
        d = safe_load_npz(GRID_STATE)
        if d is None:
            raise RuntimeError(f"No grid state found at {GRID_STATE}. Run with DO_COMPUTE=True first.")
        grid = dict(
            widths=d["widths"].astype(int).tolist(),
            depths=d["depths"].astype(int).tolist(),
            gains=d["gains"].astype(float).tolist(),
            base_lrs=d["base_lrs"].astype(float).tolist(),
            seeds=d["seeds"].astype(int).tolist(),
            G_lambda_map=d["G_lambda_map"],
            G_J_map=d["G_J_map"],
            rho_lambda_map=d["rho_lambda_map"],
            rho_J_map=d["rho_J_map"],
            done_map=d["done_map"],
        )

    if DO_PLOT:
        print("[plot] generating figures ...")
        plot_all_slices(grid)
        print("[plot] saved to:", PLOT_DIR)
