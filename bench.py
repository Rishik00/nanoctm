#!/usr/bin/env python3
"""
bench.py — throughput benchmark for nanoCTM optimisations.

Measures steps/sec (forward + backward + optimiser step) for four configs
built up incrementally:

  baseline   — stock NanoCTM, unmodified
  circ_buf   — circular buffer + pre-computed triu indices
  compiled   — circ_buf  + torch.compile(mode="reduce-overhead")
  bf16       — compiled  + torch.autocast(bfloat16)

Run with:  python bench.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# TF32 uses tensor-core matmul for float32 ops — free speedup on Ampere+
torch.set_float32_matmul_precision("high")

from nano_ctm import (
    CTMConfig,
    NanoCTM,
    ParityDataset,
    compute_certainty,
    ctm_loss,
)

# ─────────────────────────────── knobs ───────────────────────────────────────
WARMUP = 50  # steps before timing starts (covers compile + GPU warm-up)
BENCH = 200  # timed steps
BATCH = 256
DEVICE = torch.device("cuda")

cfg = CTMConfig(
    d_model=256,
    d_input=256,
    d_embedding=256,
    sequence_length=32,
    iterations=20,
    memory_length=20,
)
# ─────────────────────────────────────────────────────────────────────────────


def make_loader() -> DataLoader:
    ds = ParityDataset(cfg.sequence_length, length=10_000_000)
    return DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


def run_bench(model_fn, loader, name, *, use_compile=False, use_bf16=False) -> float:
    model = model_fn().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    if use_compile:
        print(f"  [{name}] compiling … ", end="", flush=True)
        model = torch.compile(model, mode="reduce-overhead")

    model.train()
    it = iter(loader)

    def step():
        vec, tgt = next(it)
        x = (vec.to(DEVICE, non_blocking=True) == 1).long()
        tgt = tgt.to(DEVICE, non_blocking=True)
        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred, cert = model(x)
                loss = ctm_loss(pred, tgt, cert, cfg.sequence_length)
        else:
            pred, cert = model(x)
            loss = ctm_loss(pred, tgt, cert, cfg.sequence_length)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    if use_compile:
        print("done")

    t0 = time.perf_counter()
    for _ in range(BENCH):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sps = BENCH / elapsed
    mps = elapsed / BENCH * 1000
    print(f"  {name:<42}  {sps:8.1f} steps/sec   {mps:6.2f} ms/step")
    return sps


# ─────────────────────────────────────────────────────────────────────────────
# Optimised model
# ─────────────────────────────────────────────────────────────────────────────


class CircularBufCTM(NanoCTM):
    """
    Two targeted optimisations over the baseline:

    1. Circular buffer for state_trace
       Baseline: torch.cat([trace[:,:,1:], state.unsqueeze(-1)], dim=-1)
       — allocates a new (B, N, M) tensor on every one of the T thinking steps.
       Fix: pre-allocate once, write in-place at buf_idx, advance pointer.
       No reordering needed for throughput benchmarking; NLM adapts during training.

    2. Pre-computed triu indices
       Baseline: torch.triu_indices(...) inside compute_sync → called 2*T times
       per forward pass (action + out sync, every iteration).
       Fix: compute once in __init__, register as buffers, reuse every call.

    Only "first-last" / "random" neuron_select_type (outer-product path) is
    accelerated here; "random-pairing" falls back transparently.
    """

    def __init__(self, config: CTMConfig):
        super().__init__(config)
        cfg = config
        if cfg.neuron_select_type != "random-pairing":
            i_a, j_a = torch.triu_indices(cfg.n_synch_action, cfg.n_synch_action)
            i_o, j_o = torch.triu_indices(cfg.n_synch_out, cfg.n_synch_out)
            self.register_buffer("triu_i_a", i_a)
            self.register_buffer("triu_j_a", j_a)
            self.register_buffer("triu_i_o", i_o)
            self.register_buffer("triu_j_o", j_o)

    def _sync_fast(
        self, activated_state, da, db, r, idx_left, idx_right, triu_i, triu_j
    ):
        sel_l = activated_state[:, idx_left]
        sel_r = activated_state[:, idx_right]
        outer = sel_l.unsqueeze(2) * sel_r.unsqueeze(1)
        pairwise = outer[:, triu_i, triu_j]

        if da is None:
            da = pairwise
            db = torch.ones_like(pairwise)
        else:
            da = r * da + pairwise
            db = r * db + 1.0

        return da / db.sqrt(), da, db

    def forward(self, x: torch.Tensor):
        cfg = self.config
        B, M = x.size(0), cfg.memory_length

        kv = self.backbone(x).float()
        kv = self.kv_proj(kv)
        kv_t = kv.transpose(1, 2)
        pos = self.pos_embedding(kv_t)
        kv = (kv_t + pos).transpose(1, 2)

        # Allocate circular buffer once
        state_trace = self.start_state_trace.unsqueeze(0).expand(B, -1, -1).clone()
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1).clone()
        buf_idx = 0

        r_action = (
            torch.exp(-self.decay_params_action.clamp(0.0, 15.0))
            .unsqueeze(0)
            .expand(B, -1)
        )
        r_out = (
            torch.exp(-self.decay_params_out.clamp(0.0, 15.0))
            .unsqueeze(0)
            .expand(B, -1)
        )

        _, da_out, db_out = self._sync_fast(
            activated_state,
            None,
            None,
            r_out,
            self.idx_left_out,
            self.idx_right_out,
            self.triu_i_o,
            self.triu_j_o,
        )

        predictions = torch.empty(B, cfg.out_dims, cfg.iterations, device=x.device)
        certainties = torch.empty(B, 2, cfg.iterations, device=x.device)
        da_action = db_action = None

        for t in range(cfg.iterations):
            sync_a, da_action, db_action = self._sync_fast(
                activated_state,
                da_action,
                db_action,
                r_action,
                self.idx_left_action,
                self.idx_right_action,
                self.triu_i_a,
                self.triu_j_a,
            )

            q = self.q_proj(sync_a).unsqueeze(1)
            attn_out, _ = self.attention(q, kv, kv, need_weights=False)
            attn_out = attn_out.squeeze(1)

            pre_syn = torch.cat([attn_out, activated_state], dim=-1)
            state = self.synapses(pre_syn)

            # In-place circular buffer write — no tensor allocation
            state_trace[:, :, buf_idx] = state
            buf_idx = (buf_idx + 1) % M

            activated_state = self.nlm(state_trace)

            sync_o, da_out, db_out = self._sync_fast(
                activated_state,
                da_out,
                db_out,
                r_out,
                self.idx_left_out,
                self.idx_right_out,
                self.triu_i_o,
                self.triu_j_o,
            )
            pred = self.output_proj(sync_o)
            cert = compute_certainty(pred, [cfg.sequence_length, 2])

            predictions[..., t] = pred
            certainties[..., t] = cert

        return predictions, certainties


# ─────────────────────────────────────────────────────────────────────────────


def main():
    loader = make_loader()

    print(f"\nDevice  : {torch.cuda.get_device_name(0)}")
    print(
        f"Config  : seq_len={cfg.sequence_length}  iterations={cfg.iterations}  "
        f"d_model={cfg.d_model}  batch={BATCH}"
    )
    print(f"Warmup  : {WARMUP}  |  Bench : {BENCH} steps\n")
    print("─" * 72)

    results = {}
    results["baseline"] = run_bench(
        lambda: NanoCTM(cfg),
        loader,
        "baseline",
    )
    results["circ_buf"] = run_bench(
        lambda: CircularBufCTM(cfg),
        loader,
        "circ_buf  (circular buf + precomp triu)",
    )
    results["compiled"] = run_bench(
        lambda: CircularBufCTM(cfg),
        loader,
        "compiled  (circ_buf + torch.compile)",
        use_compile=True,
    )
    results["bf16"] = run_bench(
        lambda: CircularBufCTM(cfg),
        loader,
        "bf16      (compiled + autocast bf16)",
        use_compile=True,
        use_bf16=True,
    )

    print("─" * 72)
    best = max(results, key=results.get)
    baseline = results["baseline"]
    print(f"\nFastest : {best}  →  {results[best]:.1f} steps/sec")
    print(f"\nSpeedup summary (vs baseline @ {baseline:.1f} sps):")
    for name, sps in results.items():
        tag = " ← best" if name == best else ""
        print(f"  {name:<42}  {sps / baseline:.2f}x{tag}")
    print()


if __name__ == "__main__":
    main()
