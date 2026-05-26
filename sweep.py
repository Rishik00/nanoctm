"""
sweep.py
Hyperparameter sweep over (iterations × memory_length) for NanoCTM.

Captures per-thinking-step accuracy and certainty on the validation set
so we can see exactly where the model "loses track" as iterations grow.

Two sweeps:
  A) iterations ∈ {1,5,10,20,50,100} — memory_length fixed at 20
  B) memory_length ∈ {5,10,20,50,100} — iterations fixed at 50

For speed, training is capped at MAX_STEPS per run.
"""

import sys, os, json, time, csv
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from triain_bert_nanoctm import CTMConfig, NanoCTM, SST2Dataset, ctm_loss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_STEPS = 500        # training steps per run (fast enough to see trends)
LOG_EVERY  = 100       # log interval during training
D_MODEL    = 256
D_INPUT    = 256
NUM_HEADS  = 8
N_SYNCH    = 64
DROPOUT    = 0.1

SWEEP_GRID = [
    # --- A: iterations sweep (memory fixed at 20) ---
    dict(tag="A", name="iter_001_mem_20", iterations=1,   memory_length=20),
    dict(tag="A", name="iter_005_mem_20", iterations=5,   memory_length=20),
    dict(tag="A", name="iter_010_mem_20", iterations=10,  memory_length=20),
    dict(tag="A", name="iter_020_mem_20", iterations=20,  memory_length=20),
    dict(tag="A", name="iter_050_mem_20", iterations=50,  memory_length=20),
    dict(tag="A", name="iter_100_mem_20", iterations=100, memory_length=20),
    # --- B: memory sweep (iterations fixed at 50) ---
    dict(tag="B", name="iter_050_mem_005", iterations=50, memory_length=5),
    dict(tag="B", name="iter_050_mem_010", iterations=50, memory_length=10),
    # iter_050_mem_020 already in A
    dict(tag="B", name="iter_050_mem_050", iterations=50, memory_length=50),
    dict(tag="B", name="iter_050_mem_100", iterations=50, memory_length=100),
]

# ---------------------------------------------------------------------------
# Per-step validation
# ---------------------------------------------------------------------------

def val_per_step(model, val_loader, device, T, n_classes):
    """
    Returns:
        acc_per_t  : list[float] of length T — accuracy at each thinking step
        cert_per_t : list[float] of length T — mean certainty at each thinking step
        best_t     : int — thinking step with highest accuracy
        cert_mono_pct : float — fraction of samples where certainty is monotonically
                        non-decreasing from t=0..T-1 (measure of "thinking works")
    """
    model.eval()
    # accumulators
    correct   = torch.zeros(T, device=device)
    cert_sum  = torch.zeros(T, device=device)
    mono_count = 0
    total = 0

    with torch.no_grad():
        for input_ids, attn_mask, target in val_loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            target    = target.to(device)

            preds, certs = model(input_ids, attn_mask)
            # preds: (B, C, T), certs: (B, 2, T)

            B = target.size(0)
            total += B

            for t in range(T):
                correct[t]  += (preds[:, :, t].argmax(1) == target).sum()
                cert_sum[t] += certs[:, 1, t].sum()

            # monotonicity: cert[t] >= cert[t-1] for all t>0 ?
            cert_seq = certs[:, 1, :]  # (B, T)
            mono = (cert_seq[:, 1:] >= cert_seq[:, :-1]).all(dim=1)  # (B,)
            mono_count += mono.sum().item()

    acc_per_t  = (correct / total).cpu().tolist()
    cert_per_t = (cert_sum / total).cpu().tolist()
    best_t     = int(torch.tensor(acc_per_t).argmax())
    cert_mono_pct = mono_count / total if T > 1 else 1.0

    model.train()
    return acc_per_t, cert_per_t, best_t, cert_mono_pct


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(exp, train_loader, val_loader, device, log_dir):
    name        = exp["name"]
    iterations  = exp["iterations"]
    memory_length = exp["memory_length"]

    cfg = CTMConfig(
        d_model=D_MODEL,
        d_input=D_INPUT,
        num_heads=NUM_HEADS,
        iterations=iterations,
        memory_length=memory_length,
        n_synch_out=N_SYNCH,
        n_synch_action=N_SYNCH,
        dropout=DROPOUT,
    )

    on_cuda = device.type == "cuda"
    model   = NanoCTM(cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.01,
                               fused=(on_cuda))

    trainable_params = sum(p.numel() for p in trainable)
    T = iterations

    print(f"\n{'─'*60}")
    print(f"RUN: {name}  (iter={iterations}, mem={memory_length})")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  iter/mem ratio  : {iterations/memory_length:.1f}x  "
          f"({'wrapping' if iterations > memory_length else 'no-wrap'})")

    log_path     = os.path.join(log_dir, f"{name}.log")
    metrics_path = os.path.join(log_dir, f"{name}_metrics.jsonl")

    train_log_rows = []  # per-checkpoint rows logged during training

    t0 = time.time()
    model.train()

    for step, (input_ids, attn_mask, target) in enumerate(train_loader):
        if step >= MAX_STEPS:
            break

        input_ids = input_ids.to(device, non_blocking=on_cuda)
        attn_mask = attn_mask.to(device, non_blocking=on_cuda)
        target    = target.to(device, non_blocking=on_cuda)

        if on_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds, certs = model(input_ids, attn_mask)
                loss = ctm_loss(preds, target, certs)
        else:
            preds, certs = model(input_ids, attn_mask)
            loss = ctm_loss(preds, target, certs)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()

        if step % LOG_EVERY == 0:
            acc_t  = [(preds[:, :, t].argmax(1) == target).float().mean().item() for t in range(T)]
            cert_t = [certs[:, 1, t].mean().item() for t in range(T)]
            row = {
                "step": step,
                "loss": round(loss.item(), 4),
                "acc_final": round(acc_t[-1], 4),
                "acc_t0": round(acc_t[0], 4),
                "cert_t0": round(cert_t[0], 4),
                "cert_tT": round(cert_t[-1], 4),
                "cert_max": round(max(cert_t), 4),
                "best_t_train": int(torch.tensor(acc_t).argmax()),
            }
            train_log_rows.append(row)
            print(f"  step {step:4d} | loss {row['loss']:.4f} | "
                  f"acc@final {row['acc_final']:.3f} | "
                  f"cert t0→tT {row['cert_t0']:.3f}→{row['cert_tT']:.3f} | "
                  f"cert_max {row['cert_max']:.3f} (t={row['best_t_train']})")

    # --- Validation with per-step breakdown ---
    T_actual = T
    val_acc_per_t, val_cert_per_t, best_t, cert_mono_pct = val_per_step(
        model, val_loader, device, T_actual, cfg.n_classes
    )

    elapsed = time.time() - t0
    result = {
        "name": name,
        "tag": exp["tag"],
        "iterations": iterations,
        "memory_length": memory_length,
        "iter_mem_ratio": round(iterations / memory_length, 2),
        "wrapping": iterations > memory_length,
        "trainable_params": trainable_params,
        "val_acc_final_t": round(val_acc_per_t[-1], 4),   # accuracy at last thinking step
        "val_acc_best_t":  round(max(val_acc_per_t), 4),  # best accuracy over all t
        "best_t": best_t,
        "cert_mono_pct": round(cert_mono_pct, 4),
        "val_acc_per_t":  [round(x, 4) for x in val_acc_per_t],
        "val_cert_per_t": [round(x, 4) for x in val_cert_per_t],
        "elapsed_min": round(elapsed / 60, 2),
        "train_log": train_log_rows,
    }

    # Write per-run log
    with open(log_path, "w") as f:
        f.write(f"RUN: {name}\n")
        f.write(f"Config: iter={iterations}, mem={memory_length}, d_model={D_MODEL}\n")
        f.write(f"Trainable params: {trainable_params:,}\n")
        f.write(f"Elapsed: {elapsed/60:.2f} min\n\n")
        f.write("=== Training log ===\n")
        for row in train_log_rows:
            f.write(json.dumps(row) + "\n")
        f.write("\n=== Validation per-step ===\n")
        for t in range(T_actual):
            f.write(f"  t={t:3d}: acc={val_acc_per_t[t]:.4f}  cert={val_cert_per_t[t]:.4f}\n")
        f.write(f"\nBest t: {best_t}  (val_acc={max(val_acc_per_t):.4f})\n")
        f.write(f"Val acc @ final t: {val_acc_per_t[-1]:.4f}\n")
        f.write(f"Cert monotone %: {cert_mono_pct:.3f}\n")

    print(f"\n  [Val] acc@t=T {val_acc_per_t[-1]:.4f} | "
          f"best acc {max(val_acc_per_t):.4f} at t={best_t} | "
          f"cert_mono {cert_mono_pct:.3f}")
    print(f"  Elapsed: {elapsed/60:.1f} min")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = device.type == "cuda"
    batch_size = 64 if on_cuda else 16

    print(f"Device: {device}  batch_size: {batch_size}  max_steps: {MAX_STEPS}")

    log_dir = os.path.join(os.path.dirname(__file__), "logs", "sweep")
    os.makedirs(log_dir, exist_ok=True)

    # Load dataset once, reuse across all runs
    print("\nLoading SST-2 dataset (once)...")
    backbone = "distilbert-base-uncased"
    train_dataset = SST2Dataset("train",      backbone, max_length=128)
    val_dataset   = SST2Dataset("validation", backbone, max_length=128)
    print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}\n")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=on_cuda, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=on_cuda, persistent_workers=True,
    )

    all_results = []
    for exp in SWEEP_GRID:
        result = run_one(exp, train_loader, val_loader, device, log_dir)
        all_results.append(result)

    # --- Summary table ---
    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'Name':<22} {'iter':>5} {'mem':>5} {'ratio':>6} {'wrap':>6} "
          f"{'acc@tT':>7} {'acc@best':>9} {'best_t':>7} {'cert_mono':>10}")
    print("─" * 80)
    for r in all_results:
        print(
            f"{r['name']:<22} {r['iterations']:5d} {r['memory_length']:5d} "
            f"{r['iter_mem_ratio']:6.1f} {'yes' if r['wrapping'] else 'no':>6} "
            f"{r['val_acc_final_t']:7.4f} {r['val_acc_best_t']:9.4f} "
            f"{r['best_t']:7d} {r['cert_mono_pct']:10.3f}"
        )

    # Save full results to JSON
    out_path = os.path.join(log_dir, "sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV for easy analysis
    csv_path = os.path.join(log_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "tag", "iterations", "memory_length", "iter_mem_ratio",
            "wrapping", "val_acc_final_t", "val_acc_best_t",
            "best_t", "cert_mono_pct", "elapsed_min"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"\nFull results → {out_path}")
    print(f"CSV          → {csv_path}")
    print(f"Per-run logs → {log_dir}/*.log")

    # --- Key findings ---
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")

    sweep_A = [r for r in all_results if r["tag"] == "A"]
    sweep_B = [r for r in all_results if r["tag"] == "B"]

    if sweep_A:
        best_A = max(sweep_A, key=lambda r: r["val_acc_best_t"])
        print(f"\n[Sweep A — iterations, mem=20]")
        print(f"  Best: {best_A['name']}  (iter={best_A['iterations']})  "
              f"acc={best_A['val_acc_best_t']:.4f}")
        wrap_runs = [(r['iterations'], r['val_acc_best_t']) for r in sweep_A if r['wrapping']]
        nowrap_runs = [(r['iterations'], r['val_acc_best_t']) for r in sweep_A if not r['wrapping']]
        if wrap_runs and nowrap_runs:
            avg_wrap   = sum(v for _, v in wrap_runs)   / len(wrap_runs)
            avg_nowrap = sum(v for _, v in nowrap_runs) / len(nowrap_runs)
            print(f"  Avg acc  wrapping runs: {avg_wrap:.4f}")
            print(f"  Avg acc no-wrap  runs : {avg_nowrap:.4f}")
            delta = avg_wrap - avg_nowrap
            direction = "better" if delta > 0 else "worse"
            print(f"  → Wrapping is {abs(delta)*100:.1f}pp {direction} than no-wrapping on average")

    if sweep_B:
        best_B = max(sweep_B, key=lambda r: r["val_acc_best_t"])
        print(f"\n[Sweep B — memory_length, iter=50]")
        print(f"  Best: {best_B['name']}  (mem={best_B['memory_length']})  "
              f"acc={best_B['val_acc_best_t']:.4f}")

    # cert monotonicity finding
    all_mono = [(r['iterations'], r['cert_mono_pct']) for r in all_results]
    print(f"\n[Certainty monotonicity — % of val samples where cert rises t=0→T]")
    for iters, mono in sorted(all_mono):
        bar = "█" * int(mono * 20)
        print(f"  iter={iters:3d}: {mono:.3f}  {bar}")


if __name__ == "__main__":
    main()
