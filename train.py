import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import yaml, json, time
import numpy as np

from ctm import CTM, CTMConfig

# TODO: move dataset to experiments/ once that folder exists
from legacy.nano_ctm import ParityDataset


def ctm_loss(preds: torch.Tensor, targets: torch.Tensor, certs: torch.Tensor) -> torch.Tensor:
    """Certainty-weighted cross-entropy, averaged over T thinking steps."""
    B, _, T = preds.shape
    seq_len = targets.size(1)
    # Softmax-normalise so weights always sum to 1 across steps,
    # preventing the model from escaping the loss by being maximally uncertain.
    cert_weights = F.softmax(certs[:, 1, :], dim=-1)  # (B, T)
    total = preds.new_zeros(())
    for t in range(T):
        logits = preds[:, :, t].reshape(B * seq_len, 2)
        tgts   = targets.reshape(B * seq_len)
        ce     = F.cross_entropy(logits, tgts, reduction="none").reshape(B, seq_len).mean(1)
        total  = total + (cert_weights[:, t] * ce).mean()
    return total


def make_optimizer(model: nn.Module, args: dict) -> torch.optim.Optimizer:
    trainable  = [p for p in model.parameters() if p.requires_grad]
    optim_name = args.get("optimizer", "adam")

    if optim_name == "muon":
        # Muon requires a third-party package — use 'adam' until that's added.
        raise NotImplementedError("Muon optimizer is not yet wired in; set optimizer: adam.")

    return torch.optim.AdamW(
        trainable,
        lr=args["lr"],
        weight_decay=args["weight_decay"],
        fused=torch.cuda.is_available(),
    )


def train(args: dict) -> dict:
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    t_total_start = time.perf_counter()
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    on_cuda = device == "cuda"

    log_dir = Path(args["log_dir"]) / args["run_name"]
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg = CTMConfig(
        backbone_type=args["backbone"],
        max_seq_len=args["seq_len"],
        d_model=args["d_model"],
        d_input=args["d_model"],
        n_synch_out=args["n_synch"],
        n_synch_action=args["n_synch"],
        memory_length=args["memory_length"],
        iterations=args["iterations"],
        dropout=0.1,
        use_flash=args["use_flash"],
        use_triton=args["use_triton"],
    )

    config_meta = {
        "run_name":       args["run_name"],
        "seed":           args["seed"],
        "backbone":       args["backbone"],
        "seq_len":        args["seq_len"],
        "d_model":        cfg.d_model,
        "n_synch":        cfg.n_synch_action,
        "memory_length":  cfg.memory_length,
        "iterations":     cfg.iterations,
        "d_backbone":     cfg.d_backbone,
        "synch_rep_size": cfg.synch_rep_size_action,
        "batch_size":     args["batch_size"],
        "lr":             args["lr"],
        "weight_decay":   args["weight_decay"],
        "optimizer":      args["optimizer"],
        "use_compile":    args["use_compile"],
        "use_flash":      args["use_flash"],
        "use_triton":     args["use_triton"],
    }
    (log_dir / "config.json").write_text(json.dumps(config_meta, indent=2))

    train_ds = ParityDataset(args["seq_len"], length=1_000_000)
    val_ds   = ParityDataset(args["seq_len"], length=10_000)
    print(f"  [{args['run_name']}] Train {len(train_ds):,}  Val {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args["batch_size"], shuffle=True,
        num_workers=4, pin_memory=on_cuda, persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args["batch_size"] * 2, shuffle=False,
        num_workers=2, pin_memory=on_cuda, persistent_workers=True,
    )

    # TF32 matmuls on Ampere+ GPUs: same dynamic range as float32, ~3× faster.
    if on_cuda:
        torch.set_float32_matmul_precision("high")

    raw_model = CTM(cfg).to(device)
    n_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    config_meta["trainable_params"]   = n_trainable
    config_meta["trainable_params_M"] = round(n_trainable / 1e6, 3)
    print(f"  [{args['run_name']}] Trainable params: {n_trainable:,} ({n_trainable/1e6:.2f}M)")

    model = raw_model
    if args["use_compile"] and on_cuda:
        print("Compiling model...")
        model = torch.compile(raw_model, mode=args["compile_mode"], fullgraph=False)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            dummy = torch.zeros(args["batch_size"], args["seq_len"], device=device, dtype=torch.long)
            _ = model(dummy)
        torch.cuda.synchronize()
        print("Compilation done.")

    trainable_params = [p for p in raw_model.parameters() if p.requires_grad]
    optim = make_optimizer(raw_model, args)

    records   = []
    log_fh    = open(log_dir / "train.jsonl", "w")
    max_steps = args["max_steps"]

    raw_model.train()
    t_train_start = time.perf_counter()

    for idx, (vector, target) in enumerate(train_loader):
        x      = (vector == 1).long().to(device, non_blocking=on_cuda)
        target = target.to(device, non_blocking=on_cuda)

        if on_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds, certs = model(x)
                loss = ctm_loss(preds, target, certs)
        else:
            preds, certs = model(x)
            loss = ctm_loss(preds, target, certs)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(trainable_params, 1.0).item()
        optim.step()

        with torch.no_grad():
            B, seq_len = target.shape
            pred_classes = preds[:, :, -1].reshape(B, seq_len, 2).argmax(-1)
            acc          = (pred_classes == target).float().mean().item()
            cert_start   = certs[:, 1, 0].mean().item()
            cert_final   = certs[:, 1, -1].mean().item()
            cert_profile = certs[:, 1, :].mean(0).tolist()

        rec = {
            "step":         idx,
            "loss":         round(loss.item(), 6),
            "acc":          round(acc, 6),
            "cert_t0":      round(cert_start, 6),
            "cert_tT":      round(cert_final, 6),
            "grad_norm":    round(grad_norm, 6),
            "cert_profile": [round(v, 6) for v in cert_profile],
            "elapsed_s":    round(time.perf_counter() - t_train_start, 3),
        }
        records.append(rec)
        log_fh.write(json.dumps(rec) + "\n")
        log_fh.flush()

        if idx % 100 == 0:
            elapsed = time.perf_counter() - t_train_start
            ms_per_step = (elapsed / max(idx, 1)) * 1000 if idx > 0 else 0.0
            print(f"  [{args['run_name']}] step {idx:4d} | loss {loss.item():.4f} "
                  f"| acc {acc:.3f} | cert {cert_start:.3f}→{cert_final:.3f} "
                  f"| {ms_per_step:.1f}ms/step | {elapsed:.1f}s elapsed")

        if max_steps > 0 and idx >= max_steps:
            break

    log_fh.close()
    train_elapsed = time.perf_counter() - t_train_start
    total_steps   = len(records)

    val_acc, val_loss_avg, vcert_start, vcert_final = run_eval(
        model, val_loader, device, on_cuda, args["run_name"],
    )
    total_elapsed = time.perf_counter() - t_total_start

    print(f"  [{args['run_name']}] {train_elapsed:.1f}s train | "
          f"{total_steps / train_elapsed:.1f} steps/s | "
          f"{train_elapsed / total_steps * 1000:.1f} ms/step")

    summary = {
        **config_meta,
        "total_steps":      total_steps,
        "val_acc":          round(val_acc, 6),
        "val_loss":         round(val_loss_avg, 6),
        "val_cert_t0":      round(float(np.mean(vcert_start)), 6),
        "val_cert_tT":      round(float(np.mean(vcert_final)), 6),
        "train_loss_final": records[-1]["loss"],
        "train_acc_final":  records[-1]["acc"],
        "train_elapsed_s":  round(train_elapsed, 2),
        "total_elapsed_s":  round(total_elapsed, 2),
        "ms_per_step":      round(train_elapsed / total_steps * 1000, 2),
        "steps_per_sec":    round(total_steps / train_elapsed, 2),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  [{args['run_name']}] Done → {log_dir}/")
    return summary


def run_eval(
    model, val_loader, device: str, on_cuda: bool, run_name: str
) -> tuple:
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.eval()

    correct = total_samples = 0
    batch_losses, cert_start_vals, cert_final_vals = [], [], []

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if on_cuda else torch.no_grad()
    with torch.no_grad(), ctx:
        for vector, target in val_loader:
            x      = (vector == 1).long().to(device)
            target = target.to(device)
            preds, certs = raw_model(x)

            B, seq_len = target.shape
            pred_classes = preds[:, :, -1].reshape(B, seq_len, 2).argmax(-1)
            correct       += (pred_classes == target).sum().item()
            total_samples += target.size(0)

            batch_losses.append(ctm_loss(preds, target, certs).item())
            cert_start_vals.append(certs[:, 1, 0].mean().item())
            cert_final_vals.append(certs[:, 1, -1].mean().item())

    val_acc      = correct / total_samples
    val_loss_avg = float(np.mean(batch_losses))

    print(f"  [{run_name}] Val acc={val_acc:.4f} loss={val_loss_avg:.4f} "
          f"cert {np.mean(cert_start_vals):.3f}→{np.mean(cert_final_vals):.3f}")

    return val_acc, val_loss_avg, cert_start_vals, cert_final_vals


if __name__ == "__main__":
    with open("config.yaml") as f:
        args = yaml.safe_load(f)
    train(args)
