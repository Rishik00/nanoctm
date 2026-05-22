"""
train_parity.py
---------------
Research training script for CTM on the parity task.

Supports:
  - Scaled-up configs (seq_len=1024, d_model=512+)
  - Full metric logging every step: loss, accuracy, grad norms (total + per-component),
    certainty at t=0 and t=T, step time
  - Fixed validation holdout, evaluated every --val-every steps
  - Structured log output:
      logs/<timestamp>_<name>/
          config.json    full config snapshot
          train.jsonl    one record per training step
          val.jsonl      one record per validation run
          summary.json   written at the end

  - Iteration sweep:  T ∈ {5, 10, 20, 40}
  - Scale sweep:      d_model ∈ {256, 512, 1024}
  - Memory sweep:     memory_length ∈ {20, 64, 128, 256}

Usage:
  python train_parity.py                          # single run, default config
  python train_parity.py --sweep iterations       # sweep T
  python train_parity.py --sweep scale            # sweep d_model
  python train_parity.py --sweep memory           # sweep memory_length
  python train_parity.py --d-model 1024 --steps 10000 --name big
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nano_ctm import CTMConfig, NanoCTM, ParityDataset, ctm_loss

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    # ---- Model ----
    d_model: int = 512
    d_input: int = 512
    d_embedding: int = 512
    sequence_length: int = 1024
    iterations: int = 20
    memory_length: int = 64
    n_synch_out: int = 64
    n_synch_action: int = 64
    neuron_select_type: str = "first-last"
    num_heads: int = 8
    dropout: float = 0.1

    # ---- Training ----
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    total_steps: int = 5000
    grad_clip: float = 1.0

    # ---- Validation ----
    val_every: int = 200
    val_size: int = 4096  # fixed holdout: generated once, reused throughout

    # ---- Infrastructure ----
    name: str = "parity_1024"
    compile_model: bool = True
    bf16: bool = True
    seed: int = 42


# ---------------------------------------------------------------------------
# CTMConfig from TrainConfig
# ---------------------------------------------------------------------------


def build_ctm_config(tc: TrainConfig) -> CTMConfig:
    return CTMConfig(
        d_model=tc.d_model,
        d_input=tc.d_input,
        d_embedding=tc.d_embedding,
        sequence_length=tc.sequence_length,
        iterations=tc.iterations,
        memory_length=tc.memory_length,
        n_synch_out=tc.n_synch_out,
        n_synch_action=tc.n_synch_action,
        neuron_select_type=tc.neuron_select_type,
        num_heads=tc.num_heads,
        dropout=tc.dropout,
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_parity_data(
    seq_len: int, n: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate n parity sequences of length seq_len.

    Returns:
        x:      (n, seq_len) int64 ∈ {0, 1}   — binary input (1 for +1, 0 for -1)
        labels: (n, seq_len) int64 ∈ {0, 1}   — running parity (odd # of -1s seen so far)
    """
    vec = (2 * torch.randint(0, 2, (n, seq_len), device=device) - 1).float()
    labels = (vec < 0).long().cumsum(dim=1) % 2
    x = (vec == 1).long()
    return x, labels


# ---------------------------------------------------------------------------
# Grad norm helpers
# ---------------------------------------------------------------------------

# Maps a display name to a tuple of parameter-name prefixes that belong to that group.
_GRAD_GROUPS: Dict[str, Tuple[str, ...]] = {
    "backbone": ("backbone.",),
    "kv_proj": ("kv_proj.",),
    "pos_emb": ("pos_embedding.",),
    "nlm": ("nlm.",),
    "synapses": ("synapses.",),
    "attention": ("attention.",),
    "q_proj": ("q_proj.",),
    "output_proj": ("output_proj.",),
    "decay": ("decay_params_",),
    "init_state": ("start_",),
}


def compute_grad_norms(model: nn.Module) -> Dict[str, float]:
    """
    Returns a dict with:
      "total"         — global grad norm (same value clip_grad_norm_ uses)
      "<group>"       — per-component L2 grad norm for each group in _GRAD_GROUPS

    Call this AFTER loss.backward() and BEFORE optim.step().
    """
    named_grads: Dict[str, torch.Tensor] = {
        name: p.grad for name, p in model.named_parameters() if p.grad is not None
    }

    if not named_grads:
        return {"total": 0.0, **{k: 0.0 for k in _GRAD_GROUPS}}

    total_sq = sum(g.detach().norm() ** 2 for g in named_grads.values())
    result: Dict[str, float] = {"total": total_sq.sqrt().item()}

    for group, prefixes in _GRAD_GROUPS.items():
        sq = torch.tensor(0.0)
        for name, g in named_grads.items():
            if any(name.startswith(p) for p in prefixes):
                sq = sq + g.detach().norm() ** 2
        result[group] = sq.sqrt().item()

    return result


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def compute_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, seq_len: int
) -> float:
    """
    Token-level accuracy at the LAST thinking step.

    Args:
        predictions: (B, out_dims, T) float — logits across thinking steps
        targets:     (B, seq_len)     int64 — ground truth

    Returns:
        fraction of (sample, position) pairs that are correctly classified
    """
    B = predictions.size(0)
    logits_last = predictions[:, :, -1].float()  # (B, out_dims)
    logits_last = logits_last.reshape(B, seq_len, 2)  # (B, seq_len, 2)
    pred_class = logits_last.argmax(dim=-1)  # (B, seq_len)
    return (pred_class == targets).float().mean().item()


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def make_log_dir(name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"{ts}_{name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    raw_model: nn.Module,
    val_x: torch.Tensor,
    val_labels: torch.Tensor,
    ctm_cfg: CTMConfig,
    use_bf16: bool,
    chunk: int = 256,
) -> Dict[str, float]:
    """
    Run the raw (uncompiled) model over the fixed validation set in chunks.
    Returns averaged val_loss, val_acc, cert_t0, cert_tT.
    """
    raw_model.eval()
    losses: List[float] = []
    accs: List[float] = []
    c0s: List[float] = []
    cTs: List[float] = []

    for i in range(0, len(val_x), chunk):
        xb = val_x[i : i + chunk]
        yb = val_labels[i : i + chunk]

        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds, certs = raw_model(xb)
                loss = ctm_loss(preds, yb, certs, ctm_cfg.sequence_length)
        else:
            preds, certs = raw_model(xb)
            loss = ctm_loss(preds, yb, certs, ctm_cfg.sequence_length)

        losses.append(loss.item())
        accs.append(compute_accuracy(preds.float(), yb, ctm_cfg.sequence_length))
        c0s.append(certs[:, 1, 0].mean().item())
        cTs.append(certs[:, 1, -1].mean().item())

    raw_model.train()

    n = len(losses)
    return {
        "val_loss": sum(losses) / n,
        "val_acc": sum(accs) / n,
        "cert_t0": sum(c0s) / n,
        "cert_tT": sum(cTs) / n,
    }


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------


def train_run(tc: TrainConfig, log_dir: Optional[Path] = None) -> Path:
    """
    Execute one training run.  All metrics are written to log_dir.
    Returns the log_dir path.
    """
    torch.manual_seed(tc.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = device.type == "cuda"
    use_bf16 = tc.bf16 and on_cuda

    if log_dir is None:
        log_dir = make_log_dir(tc.name)

    ctm_cfg = build_ctm_config(tc)

    # --- Save config snapshot ---
    with open(log_dir / "config.json", "w") as f:
        json.dump(
            {"train": asdict(tc), "ctm": asdict(ctm_cfg)},
            f,
            indent=2,
        )

    # --- Fixed validation holdout (generated once, lives on device) ---
    val_x, val_labels = make_parity_data(tc.sequence_length, tc.val_size, device)

    # --- Dataset + loader ---
    dataset = ParityDataset(tc.sequence_length, length=10_000_000)
    loader = DataLoader(
        dataset,
        batch_size=tc.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=on_cuda,
        persistent_workers=True,
    )

    # --- Model ---
    raw_model = NanoCTM(ctm_cfg).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())

    model = raw_model
    if tc.compile_model and on_cuda:
        model = torch.compile(raw_model, mode="reduce-overhead")

    # --- Optimiser ---
    optim = torch.optim.AdamW(
        raw_model.parameters(),
        lr=tc.lr,
        weight_decay=tc.weight_decay,
        fused=on_cuda,
    )

    # --- Header ---
    print(
        f"\n{'=' * 70}\n"
        f"  {tc.name}\n"
        f"  {n_params:,} params | seq={tc.sequence_length} | "
        f"T={tc.iterations} | M={tc.memory_length} | d={tc.d_model}\n"
        f"  bf16={use_bf16}  compile={tc.compile_model and on_cuda}  "
        f"B={tc.batch_size}  steps={tc.total_steps}\n"
        f"  logs → {log_dir}\n"
        f"{'=' * 70}"
    )

    train_log = log_dir / "train.jsonl"
    val_log = log_dir / "val.jsonl"

    t_run_start = time.perf_counter()
    step = 0

    for vector, target in loader:
        if step >= tc.total_steps:
            break

        x = (vector == 1).long().to(device, non_blocking=on_cuda)
        target = target.to(device, non_blocking=on_cuda)

        t_step = time.perf_counter()

        # Forward + loss
        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds, certs = model(x)
                loss = ctm_loss(preds, target, certs, ctm_cfg.sequence_length)
        else:
            preds, certs = model(x)
            loss = ctm_loss(preds, target, certs, ctm_cfg.sequence_length)

        # Backward
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(raw_model.parameters(), tc.grad_clip)

        # Grad norms — every step, after clip (so they reflect pre-clip norms)
        gnorms = compute_grad_norms(raw_model)

        optim.step()

        step_ms = (time.perf_counter() - t_step) * 1000

        # Metrics
        loss_val = loss.item()
        acc = compute_accuracy(preds.detach().float(), target, ctm_cfg.sequence_length)
        c0 = certs[:, 1, 0].mean().item()
        cT = certs[:, 1, -1].mean().item()

        # Write train record
        record: dict = {
            "step": step,
            "loss": round(loss_val, 6),
            "acc": round(acc, 6),
            "cert_t0": round(c0, 5),
            "cert_tT": round(cT, 5),
            "step_ms": round(step_ms, 1),
            **{f"gnorm_{k}": round(v, 6) for k, v in gnorms.items()},
        }
        _append_jsonl(train_log, record)

        # Stdout — every 100 steps
        if step % 100 == 0:
            elapsed = time.perf_counter() - t_run_start
            print(
                f"  step {step:5d} | loss {loss_val:.4f} | acc {acc:.4f} "
                f"| cert {c0:.3f}→{cT:.3f} "
                f"| gnorm {gnorms['total']:.3f} | {step_ms:.0f}ms"
            )

        # Validation
        if step % tc.val_every == 0:
            vm = validate(raw_model, val_x, val_labels, ctm_cfg, use_bf16)
            val_record = {"step": step, **{k: round(v, 6) for k, v in vm.items()}}
            _append_jsonl(val_log, val_record)
            print(
                f"  [VAL]  step {step:5d} | loss {vm['val_loss']:.4f} "
                f"| acc {vm['val_acc']:.4f} "
                f"| cert {vm['cert_t0']:.3f}→{vm['cert_tT']:.3f}"
            )

        step += 1

    # --- Summary ---
    total_time = time.perf_counter() - t_run_start
    summary = {
        "name": tc.name,
        "n_params": n_params,
        "total_steps": step,
        "total_time_s": round(total_time, 1),
        "steps_per_sec": round(step / total_time, 2),
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n  Done — {step} steps in {total_time:.0f}s "
        f"({step / total_time:.1f} steps/s)\n"
    )

    return log_dir


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def sweep_iterations(base: TrainConfig) -> None:
    """How much does extra thinking time help? Sweep T ∈ {5, 10, 20, 40}."""
    for T in [5, 10, 20, 40]:
        cfg = replace(base, iterations=T, name=f"{base.name}_iter_T{T}")
        train_run(cfg)


def sweep_scale(base: TrainConfig) -> None:
    """Scaling law data. Sweep d_model ∈ {256, 512, 1024}."""
    for d in [256, 512, 1024]:
        cfg = replace(
            base,
            d_model=d,
            d_input=d,
            d_embedding=d,
            name=f"{base.name}_scale_d{d}",
        )
        train_run(cfg)


def sweep_memory(base: TrainConfig) -> None:
    """How long does each neuron need to remember? Sweep M ∈ {20, 64, 128, 256}."""
    for M in [20, 64, 128, 256]:
        cfg = replace(base, memory_length=M, name=f"{base.name}_mem_M{M}")
        train_run(cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CTM parity research training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-input", type=int, default=None, help="defaults to --d-model")
    p.add_argument(
        "--d-embedding", type=int, default=None, help="defaults to --d-model"
    )
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--memory-length", type=int, default=64)
    p.add_argument("--n-synch", type=int, default=64)
    p.add_argument(
        "--neuron-select",
        type=str,
        default="first-last",
        choices=["first-last", "random", "random-pairing"],
    )
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Validation
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--val-size", type=int, default=4096)

    # Infrastructure
    p.add_argument("--name", type=str, default="parity_1024")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # Sweep
    p.add_argument(
        "--sweep",
        choices=["iterations", "scale", "memory"],
        default=None,
        help="Run a sweep instead of a single training run",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    d = args.d_model
    base = TrainConfig(
        d_model=d,
        d_input=args.d_input or d,
        d_embedding=args.d_embedding or d,
        sequence_length=args.seq_len,
        iterations=args.iterations,
        memory_length=args.memory_length,
        n_synch_out=args.n_synch,
        n_synch_action=args.n_synch,
        neuron_select_type=args.neuron_select,
        num_heads=args.num_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=args.steps,
        grad_clip=args.grad_clip,
        val_every=args.val_every,
        val_size=args.val_size,
        name=args.name,
        compile_model=not args.no_compile,
        bf16=not args.no_bf16,
        seed=args.seed,
    )

    if args.sweep == "iterations":
        sweep_iterations(base)
    elif args.sweep == "scale":
        sweep_scale(base)
    elif args.sweep == "memory":
        sweep_memory(base)
    else:
        train_run(base)


if __name__ == "__main__":
    main()
