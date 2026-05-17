#!/usr/bin/env python3
"""
bench2.py — deep-dive benchmark targeting the three hotspots found by profiling:

  1. Attention  (27% of CUDA) — Flash Attention is sub-optimal for seq_len=32,
                                  single-query case (Q shape: B,H,1,d_head).
                                  Replace with a fused bmm path + hoist K/V.
  2. K/V projection (inside MHA) — recomputed T=20× per step even though kv
                                    is constant across the thinking loop.
                                    Hoist outside the loop: T matmuls → 1.
  3. Outer-product sync (2.65ms) — forms (B, n_synch, n_synch) then indexes
                                    triu. Replace with direct pairwise gather:
                                    state[:, il] * state[:, ir] → (B, rep).

Also tests:
  - Pre-initialised decay accumulators (removes None guard from compile graph)
  - torch.compile max-autotune
  - Batch size scaling (256 → 512 → 1024)
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nano_ctm import CTMConfig, NanoCTM, ParityDataset, compute_certainty, ctm_loss
from bench import CircularBufCTM

torch.set_float32_matmul_precision("high")

# ─────────────────────────────── knobs ───────────────────────────────────────
WARMUP = 50
BENCH  = 200
DEVICE = torch.device("cuda")

cfg = CTMConfig(
    d_model=256, d_input=256, d_embedding=256,
    sequence_length=32, iterations=20, memory_length=20,
)
# ─────────────────────────────────────────────────────────────────────────────


def make_loader(batch: int) -> DataLoader:
    ds = ParityDataset(cfg.sequence_length, length=10_000_000)
    return DataLoader(
        ds, batch_size=batch, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )


def run_bench(model_fn, batch, name, *, use_compile=True, compile_mode="reduce-overhead",
              use_bf16=True, fused_optim=False) -> float:
    loader = make_loader(batch)
    model  = model_fn().to(DEVICE)
    opt    = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.01,
        fused=fused_optim,
    )

    if use_compile:
        print(f"  [{name}] compiling ({compile_mode}) … ", end="", flush=True)
        model = torch.compile(model, mode=compile_mode)

    model.train()
    it = iter(loader)

    def step():
        vec, tgt = next(it)
        x   = (vec.to(DEVICE, non_blocking=True) == 1).long()
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
    tokens_sec = sps * batch * cfg.sequence_length / 1e6
    print(f"  {name:<52}  {sps:8.1f} sps   {mps:6.2f} ms/step   {tokens_sec:.2f}M tok/s")
    return sps


# ─────────────────────────────────────────────────────────────────────────────
# DeepOptCTM
# ─────────────────────────────────────────────────────────────────────────────

class DeepOptCTM(NanoCTM):
    """
    Stacks all targeted optimisations:

    1. KV hoisted out of the T loop
       K = attn_k_proj(kv) and V = attn_v_proj(kv) are constant across all
       T thinking steps. Compute them once before the loop; save 19/20 of
       their forward+backward cost.

    2. Manual attention (bmm path)
       For Q shape (B*H, 1, d_head) and K shape (B*H, seq=32, d_head),
       Flash Attention carries more overhead than a fused bmm pair.
       Manual: scores = Q·Kᵀ → softmax → out = scores·V. With compile,
       these two bmms fuse cleanly with surrounding elementwise ops.

    3. Direct pairwise sync (no outer product)
       Current: outer = sel_l[:,:,None] * sel_r[:,None,:] → (B, n, n) →
                triu index → (B, rep)
       New:     pairwise = state[:, il_combined] * state[:, ir_combined]
       The (B, n, n) intermediate vanishes; one fused index+mul instead.

    4. Pre-initialised decay accumulators
       Replace None guards (which create Python-level conditionals inside the
       compiled loop) with pre-zeroed tensors. Numerically identical:
       da=0, db=0 → first update: da=p, db=1 — same as the None branch.

    Inherits circular buffer + pre-computed triu indices from CircularBufCTM
    (via independent __init__ replication — avoids MRO tangles with buffers).
    """

    def __init__(self, config: CTMConfig):
        super().__init__(config)          # NanoCTM.__init__
        cfg = config
        H       = cfg.num_heads
        hd      = cfg.d_input // H

        # ── replace MHA with separate K / V / out projections ──
        # (self.attention is inherited but bypassed in forward)
        self.attn_k = nn.Linear(cfg.d_input, cfg.d_input, bias=False)
        self.attn_v = nn.Linear(cfg.d_input, cfg.d_input, bias=False)
        self.attn_o = nn.Linear(cfg.d_input, cfg.d_input, bias=True)
        self._H  = H
        self._hd = hd

        # ── pre-computed triu indices ──
        i_a, j_a = torch.triu_indices(cfg.n_synch_action, cfg.n_synch_action)
        i_o, j_o = torch.triu_indices(cfg.n_synch_out,    cfg.n_synch_out)
        self.register_buffer("triu_i_a", i_a)
        self.register_buffer("triu_j_a", j_a)
        self.register_buffer("triu_i_o", i_o)
        self.register_buffer("triu_j_o", j_o)

        # ── combined neuron indices for direct pairwise ──
        # For outer-product modes: expand via triu indices → (synch_rep_size,) index vectors.
        # For random-pairing: idx_left/right already have n_synch elements — use directly.
        if cfg.neuron_select_type != "random-pairing":
            il_a = self.idx_left_action[i_a];   ir_a = self.idx_right_action[j_a]
            il_o = self.idx_left_out[i_o];       ir_o = self.idx_right_out[j_o]
        else:
            il_a = self.idx_left_action;         ir_a = self.idx_right_action
            il_o = self.idx_left_out;            ir_o = self.idx_right_out
        self.register_buffer("sync_il_a", il_a)
        self.register_buffer("sync_ir_a", ir_a)
        self.register_buffer("sync_il_o", il_o)
        self.register_buffer("sync_ir_o", ir_o)

    # ------------------------------------------------------------------
    def _sync(self, state, da, db, r, il, ir):
        """Direct pairwise sync — no (B, n, n) intermediate."""
        pairwise = state[:, il] * state[:, ir]       # (B, synch_rep)
        da = r * da + pairwise
        db = r * db + 1.0
        return da / db.sqrt(), da, db

    # ------------------------------------------------------------------
    def _attn(self, q_vec: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              B: int) -> torch.Tensor:
        """
        Manual bmm attention for Q=(B,1,d_input), K/V=(B,seq,d_input).
        Avoids Flash Attention overhead at seq_len=32.
        """
        H, hd = self._H, self._hd
        scale = hd ** -0.5

        q = q_vec.view(B * H, 1, hd)       # (B*H, 1, hd)
        # k, v already in (B, H, seq, hd) — reshape to (B*H, seq, hd)
        kt = k.reshape(B * H, -1, hd).transpose(1, 2)   # (B*H, hd, seq)
        vt = v.reshape(B * H, -1, hd)                    # (B*H, seq, hd)

        scores = torch.bmm(q, kt) * scale                # (B*H, 1, seq)
        attn   = scores.softmax(dim=-1)
        out    = torch.bmm(attn, vt)                     # (B*H, 1, hd)
        return out.view(B, cfg.d_input)                  # (B, d_input)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        cfg = self.config
        B, M = x.size(0), cfg.memory_length
        H, hd = self._H, self._hd

        # ── encode ──
        kv = self.backbone(x).float()
        kv = self.kv_proj(kv)
        kv = (kv.transpose(1, 2) + self.pos_embedding(kv.transpose(1, 2))).transpose(1, 2)

        # ── hoist K / V projections (once, not T times) ──
        k = self.attn_k(kv).view(B, -1, H, hd).transpose(1, 2)  # (B, H, seq, hd)
        v = self.attn_v(kv).view(B, -1, H, hd).transpose(1, 2)  # (B, H, seq, hd)

        # ── recurrent state ──
        state_trace     = self.start_state_trace.unsqueeze(0).expand(B, -1, -1).clone()
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1).clone()
        buf_idx = 0

        r_action = torch.exp(-self.decay_params_action.clamp(0., 15.)).unsqueeze(0).expand(B, -1)
        r_out    = torch.exp(-self.decay_params_out.clamp(0., 15.)).unsqueeze(0).expand(B, -1)

        # ── pre-initialise accumulators (zeros ≡ first-step None init) ──
        da_a = torch.zeros(B, cfg.synch_rep_size_action, device=x.device, dtype=x.dtype if not x.is_floating_point() else torch.float32)
        db_a = torch.zeros_like(da_a)
        da_o = torch.zeros(B, cfg.synch_rep_size_out, device=x.device, dtype=da_a.dtype)
        db_o = torch.zeros_like(da_o)

        # warm-up out-sync (one step before the loop, same as baseline)
        _, da_o, db_o = self._sync(activated_state, da_o, db_o, r_out,
                                   self.sync_il_o, self.sync_ir_o)

        predictions = torch.empty(B, cfg.out_dims, cfg.iterations, device=x.device)
        certainties = torch.empty(B, 2,            cfg.iterations, device=x.device)

        for t in range(cfg.iterations):
            sync_a, da_a, db_a = self._sync(activated_state, da_a, db_a, r_action,
                                             self.sync_il_a, self.sync_ir_a)

            q_vec    = self.q_proj(sync_a)                           # (B, d_input)
            attn_out = self._attn(q_vec, k, v, B)                    # (B, d_input)
            attn_out = self.attn_o(attn_out)

            state = self.synapses(torch.cat([attn_out, activated_state], dim=-1))

            state_trace[:, :, buf_idx] = state
            buf_idx = (buf_idx + 1) % M

            activated_state = self.nlm(state_trace)

            sync_o, da_o, db_o = self._sync(activated_state, da_o, db_o, r_out,
                                             self.sync_il_o, self.sync_ir_o)

            pred = self.output_proj(sync_o)
            cert = compute_certainty(pred, [cfg.sequence_length, 2])
            predictions[..., t] = pred
            certainties[..., t] = cert

        return predictions, certainties


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxautotune", action="store_true",
                        help="include max-autotune compile (slow, ~5 min)")
    args = parser.parse_args()

    # random-pairing config: synch_rep 2080 → 64 (fastest for throughput benchmarking)
    cfg_rp = CTMConfig(
        d_model=256, d_input=256, d_embedding=256,
        sequence_length=cfg.sequence_length, iterations=cfg.iterations,
        memory_length=cfg.memory_length,
        n_synch_out=64, n_synch_action=64,
        neuron_select_type="random-pairing",
    )

    print(f"\nDevice  : {torch.cuda.get_device_name(0)}")
    print(f"Warmup  : {WARMUP}  |  Bench : {BENCH} steps")
    print(f"synch_rep  first-last={cfg.synch_rep_size_action}  "
          f"random-pairing={cfg_rp.synch_rep_size_action}\n")

    # results maps key → (sps, batch_size)
    results: dict[str, tuple[float, int]] = {}

    def rec(key, sps, B):
        results[key] = (sps, B)

    def tok(key):
        sps, B = results[key]
        return sps * B * cfg.sequence_length / 1e6

    # ── incremental stack (B=256) ──
    print("── incremental optimisation stack (B=256) ─────────────────────────")
    rec("NanoCTM+fl+B256  [eager baseline]", run_bench(
        lambda: NanoCTM(cfg), 256, "NanoCTM+first-last+B=256  [eager baseline]",
        use_compile=False, use_bf16=False, fused_optim=False,
    ), 256)
    rec("NanoCTM+fl+B256+compile", run_bench(
        lambda: NanoCTM(cfg), 256, "+compile(reduce-overhead)",
        use_compile=True, use_bf16=False, fused_optim=False,
    ), 256)
    rec("NanoCTM+fl+B256+compile+bf16", run_bench(
        lambda: NanoCTM(cfg), 256, "+bfloat16",
        use_compile=True, use_bf16=True, fused_optim=False,
    ), 256)
    rec("NanoCTM+fl+B256+compile+bf16+fused", run_bench(
        lambda: NanoCTM(cfg), 256, "+fused AdamW",
        use_compile=True, use_bf16=True, fused_optim=True,
    ), 256)

    # ── scale batch + deep optimisations ──
    print("\n── scale batch + deep optimisations ───────────────────────────────")
    rec("NanoCTM+fl+B2048+compile+bf16+fused", run_bench(
        lambda: NanoCTM(cfg), 2048, "NanoCTM+first-last+B=2048",
        use_compile=True, use_bf16=True, fused_optim=True,
    ), 2048)
    rec("DeepOpt+fl+B2048+compile+bf16+fused", run_bench(
        lambda: DeepOptCTM(cfg), 2048, "DeepOpt+first-last+B=2048  (+KV hoist)",
        use_compile=True, use_bf16=True, fused_optim=True,
    ), 2048)
    rec("NanoCTM+rp+B2048+compile+bf16+fused", run_bench(
        lambda: NanoCTM(cfg_rp), 2048, "NanoCTM+random-pair+B=2048",
        use_compile=True, use_bf16=True, fused_optim=True,
    ), 2048)
    rec("DeepOpt+rp+B2048+compile+bf16+fused", run_bench(
        lambda: DeepOptCTM(cfg_rp), 2048,
        "DeepOpt+random-pair+B=2048  [FASTEST]",
        use_compile=True, use_bf16=True, fused_optim=True,
    ), 2048)

    if args.maxautotune:
        rec("DeepOpt+rp+B2048+maxautotune", run_bench(
            lambda: DeepOptCTM(cfg_rp), 2048,
            "DeepOpt+random-pair+B=2048+max-autotune",
            use_compile=True, compile_mode="max-autotune",
            use_bf16=True, fused_optim=True,
        ), 2048)

    # ── summary (rank by tok/s, not sps) ─────────────────────────────────────
    base_tok = tok("NanoCTM+fl+B256  [eager baseline]")
    best_key = max(results, key=tok)
    print("\n" + "─" * 72)
    print(f"Eager baseline : {tok('NanoCTM+fl+B256  [eager baseline]'):.2f}M tok/s")
    print(f"Fastest        : {best_key}")
    print(f"               : {tok(best_key):.2f}M tok/s  "
          f"({tok(best_key)/base_tok:.0f}× over baseline)")
    print()


if __name__ == "__main__":
    main()
