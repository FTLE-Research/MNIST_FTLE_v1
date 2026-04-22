import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import math
import random
from contextlib import nullcontext
from typing import Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler

try:
    from torchvision.datasets import MNIST
except Exception as e:
    raise RuntimeError("This module needs torchvision installed to load MNIST.") from e

# -------------------- CONFIG --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE_TRAIN = 8192
TRAIN_ACC_TARGET = 0.90
MAX_EPOCHS = 20000
EVAL_EVERY_EPOCHS = 5
LOG_EVERY_EPOCHS = 25
SEED_BASE = 0

CHECKPOINT_DIR = "rk_ckpts_mnist"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

INPUT_DIM = 28 * 28
NUM_CLASSES = 10

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

USE_AMP_TRAIN = (DEVICE.type == "cuda")
AMP_DTYPE = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
USE_SCALER = USE_AMP_TRAIN and (DEVICE.type == "cuda") and (AMP_DTYPE == torch.float16)
SCALER = None
if USE_SCALER:
    try:
        from torch.amp import GradScaler
        SCALER = GradScaler("cuda")
    except Exception:
        SCALER = torch.cuda.amp.GradScaler()


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _amp_ctx():
    if USE_AMP_TRAIN and DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return nullcontext()


def fmt_float(x: float) -> str:
    s = f"{x:.3g}".replace(".", "p")
    return ("m" + s[1:]) if s.startswith("-") else s


def ckpt_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    return os.path.join(
        CHECKPOINT_DIR,
        f"model_N{N}_L{L}_g{fmt_float(gain)}_lr{fmt_float(base_lr)}_seed{seed}.pt",
    )


def load_or_make_mnist_data(
    cache_path: str,
    seed: int = 0,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    digits: Optional[Sequence[int]] = None,
    center: bool = True,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns flattened tensors:
      (x_train [N,784], y_train [N]), (x_test [M,784], y_test [M])
    Inputs are in [0,1] if center=False, else shifted to [-0.5,0.5].
    If digits is not None, keeps only those classes (no relabeling for multiclass).
    """
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=False)
        xt = torch.tensor(d["xt"], dtype=torch.float32)
        yt = torch.tensor(d["yt"], dtype=torch.long)
        xe = torch.tensor(d["xe"], dtype=torch.float32)
        ye = torch.tensor(d["ye"], dtype=torch.long)
        return (xt, yt), (xe, ye)

    train_ds = MNIST(root="data", train=True, download=True)
    test_ds = MNIST(root="data", train=False, download=True)

    xt = train_ds.data.float() / 255.0
    xe = test_ds.data.float() / 255.0
    if center:
        xt = xt - 0.5
        xe = xe - 0.5
    yt = train_ds.targets.long()
    ye = test_ds.targets.long()

    xt = xt.view(-1, INPUT_DIM)
    xe = xe.view(-1, INPUT_DIM)

    if digits is not None:
        dig = torch.tensor(list(digits), dtype=torch.long)
        tr_mask = (yt[..., None] == dig[None, :]).any(dim=1)
        te_mask = (ye[..., None] == dig[None, :]).any(dim=1)
        xt, yt = xt[tr_mask], yt[tr_mask]
        xe, ye = xe[te_mask], ye[te_mask]

    g = torch.Generator(device="cpu").manual_seed(seed)
    if train_limit is not None and train_limit < xt.shape[0]:
        idx = torch.randperm(xt.shape[0], generator=g)[:train_limit]
        xt, yt = xt[idx], yt[idx]
    if test_limit is not None and test_limit < xe.shape[0]:
        idx = torch.randperm(xe.shape[0], generator=g)[:test_limit]
        xe, ye = xe[idx], ye[idx]

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        np.savez(
            f,
            xt=xt.cpu().numpy().astype(np.float32),
            yt=yt.cpu().numpy().astype(np.int64),
            xe=xe.cpu().numpy().astype(np.float32),
            ye=ye.cpu().numpy().astype(np.int64),
        )

    return (xt, yt), (xe, ye)


class FC(nn.Module):
    def __init__(self, width: int, depth: int, gain: float = 1.0, input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.depth = depth
        self.gain = gain
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        prev = input_dim
        for _ in range(depth):
            l = nn.Linear(prev, width)
            nn.init.normal_(l.weight, 0.0, gain / math.sqrt(prev))
            nn.init.zeros_(l.bias)
            layers.append(l)
            prev = width
        self.hid = nn.ModuleList(layers)
        self.out = nn.Linear(prev, num_classes)
        nn.init.normal_(self.out.weight, 0.0, gain / math.sqrt(prev))
        nn.init.zeros_(self.out.bias)

    def forward(self, x, *, hid: bool = False, grad: bool = False):
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)
        for l in self.hid:
            x = torch.tanh(l(x))
        if hid:
            return x
        logits = self.out(x)
        return logits


def per_layer_lr(layer: nn.Linear, base_lr: float) -> float:
    return base_lr / float(layer.weight.size(1))


def make_optim(net: nn.Module, base_lr: float) -> torch.optim.Optimizer:
    groups = []
    for m in net.modules():
        if isinstance(m, nn.Linear):
            lr = per_layer_lr(m, base_lr)
            params = [m.weight]
            if m.bias is not None:
                params.append(m.bias)
            groups.append({"params": params, "lr": lr})
    return torch.optim.SGD(groups, momentum=0.0)


def dataset_to_loader(pair, batch_size: int, shuffle: bool, device=DEVICE) -> DataLoader:
    x, y = pair
    x = x.to(device, non_blocking=False)
    y = y.to(device, non_blocking=False)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


@torch.inference_mode()
def _acc_loss_on_tensor(net: FC, X: torch.Tensor, y: torch.Tensor, max_batch: int = 65536) -> Tuple[float, float]:
    net.eval()
    n = X.shape[0]
    correct = 0
    loss_sum = 0.0
    for s in range(0, n, max_batch):
        logits = net(X[s:s + max_batch])
        loss_sum += F.cross_entropy(logits, y[s:s + max_batch], reduction="sum").item()
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y[s:s + max_batch]).sum().item()
    return correct / float(n), loss_sum / float(n)


@torch.inference_mode()
def loader_acc_and_loss(net: FC, loader: DataLoader, _unused_loss: Optional[nn.Module] = None) -> Tuple[float, float]:
    if isinstance(loader.dataset, TensorDataset):
        X, y = loader.dataset.tensors
        return _acc_loss_on_tensor(net, X, y)
    net.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for xb, yb in loader:
        logits = net(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        pred = torch.argmax(logits, dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / float(total), loss_sum / float(total)


def train_until_acc(net: FC, train_loader: DataLoader, acc_target: float, max_epochs: int, base_lr: float) -> float:
    opt = make_optim(net, base_lr)
    X, y = train_loader.dataset.tensors if isinstance(train_loader.dataset, TensorDataset) else (None, None)
    bs = int(train_loader.batch_size or (X.shape[0] if X is not None else 8192))
    do_shuffle = isinstance(getattr(train_loader, "sampler", None), RandomSampler)

    acc = 0.0
    loss_eval = float("nan")
    for ep in range(1, max_epochs + 1):
        net.train()
        if X is not None:
            n = X.shape[0]
            perm = torch.randperm(n, device=X.device) if do_shuffle else None
            for s in range(0, n, bs):
                idx = perm[s:s + bs] if perm is not None else slice(s, s + bs)
                xb = X[idx]
                yb = y[idx]
                opt.zero_grad(set_to_none=True)
                with _amp_ctx():
                    logits = net(xb)
                    loss = F.cross_entropy(logits.float(), yb)
                if SCALER is not None:
                    SCALER.scale(loss).backward()
                    SCALER.step(opt)
                    SCALER.update()
                else:
                    loss.backward()
                    opt.step()
        else:
            for xb, yb in train_loader:
                opt.zero_grad(set_to_none=True)
                with _amp_ctx():
                    logits = net(xb)
                    loss = F.cross_entropy(logits.float(), yb)
                if SCALER is not None:
                    SCALER.scale(loss).backward()
                    SCALER.step(opt)
                    SCALER.update()
                else:
                    loss.backward()
                    opt.step()

        if ep == 1 or ep % EVAL_EVERY_EPOCHS == 0 or ep == max_epochs:
            acc, loss_eval = loader_acc_and_loss(net, train_loader)
            if ep == 1 or ep % LOG_EVERY_EPOCHS == 0 or ep == max_epochs:
                print(f"[train] epoch={ep:4d}  acc={acc:.3f}  loss={loss_eval:.4f}")
            if acc >= acc_target:
                print(f"[early-stop] hit acc_target={acc_target:.3f} at epoch {ep} (acc={acc:.3f})")
                break
    return float(acc)


def verify_or_train_checkpoint(
    N: int, L: int,
    gain: float, base_lr: float,
    seed: int,
    train_loader: DataLoader,
    acc_target: float,
    max_epochs: int,
    fail_policy: str = "return",  # return | none | raise
) -> Optional[nn.Module]:
    if fail_policy not in ("return", "none", "raise"):
        raise ValueError("fail_policy must be one of: 'return', 'none', 'raise'")

    path = ckpt_path(N, L, gain, base_lr, seed)
    net = FC(N, L, gain=gain).to(DEVICE)

    if os.path.exists(path):
        state = torch.load(path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            net.load_state_dict(state["state_dict"])
            if bool(state.get("failed", False)) and float(state.get("acc_target", acc_target)) >= acc_target:
                print(f"[ckpt] {os.path.basename(path)} previously marked FAILED (train_acc={state.get('train_acc', 'NA')})")
                net.eval()
                if fail_policy == "none":
                    return None
                if fail_policy == "raise":
                    raise RuntimeError(f"Checkpoint previously failed: {path}")
                return net
            if (not bool(state.get("failed", False))) and ("train_acc" in state) and float(state["train_acc"]) >= acc_target:
                print(f"[ckpt] loaded {os.path.basename(path)} (meta train_acc={float(state['train_acc']):.3f} ≥ target)")
                net.eval()
                return net
        else:
            net.load_state_dict(state)

        acc, _ = loader_acc_and_loss(net, train_loader)
        if acc >= acc_target:
            torch.save({
                "state_dict": net.state_dict(),
                "train_acc": float(acc),
                "acc_target": float(acc_target),
                "max_epochs": int(max_epochs),
                "failed": False,
            }, path)
            print(f"[ckpt] loaded {os.path.basename(path)} (verified acc={acc:.3f} ≥ target)")
            net.eval()
            return net
        print(f"[ckpt] {os.path.basename(path)} acc={acc:.3f} < target → continuing training")

    _seed_all(SEED_BASE + seed)
    _ = train_until_acc(net, train_loader, acc_target, max_epochs, base_lr)
    acc, _ = loader_acc_and_loss(net, train_loader)

    failed = bool(acc < acc_target)
    torch.save({
        "state_dict": net.state_dict(),
        "train_acc": float(acc),
        "acc_target": float(acc_target),
        "max_epochs": int(max_epochs),
        "failed": failed,
    }, path)

    if failed:
        print(f"[ckpt-failed] {os.path.basename(path)} train_acc={acc:.3f} < target={acc_target:.3f} → SKIPPING")
        net.eval()
        if fail_policy == "none":
            return None
        if fail_policy == "raise":
            raise RuntimeError(f"Training failed to reach acc_target for: {path}")
        return net

    print(f"[ckpt] saved {os.path.basename(path)} (train_acc={acc:.3f} ≥ target)")
    net.eval()
    return net
