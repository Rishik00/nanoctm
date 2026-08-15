> ⚠️ **LEGACY REPORT — not reproducible from current `main`.** Duplicate of `docs/reports/REPORT.md`.
> Describes the pre-refactor code (`nano_ctm.py`, `bench2.py`, `DeepOptCTM`, `neuron_select_type="random-pairing"`), now under `legacy/`. The refactored `ctm.py` on `main` is the *un-optimized baseline*. Kept as history.

# nanoCTM Benchmarking Report

**Hardware:** NVIDIA GeForce RTX 5090  
**Software:** PyTorch 2.10.0+cu130, Python 3.12  
**Model:** NanoCTM — Continuous Thought Machine, parity task  
**Metric:** steps/sec (fwd + bwd + optimiser), M tok/s (= sps × B × seq_len / 1e6)

---

## Baseline

The starting point is `nano_ctm.py` as written: eager PyTorch, float32, standard AdamW, batch size 256.

**Default config (`CTMConfig()`):**

| Field | Value | Notes |
|---|---|---|
| `d_model` | 256 | neurons / internal state size |
| `d_input` | 256 | attention Q/K/V dimension |
| `d_embedding` | 256 | backbone embedding dim |
| `num_heads` | 8 | |
| `memory_length` | 20 | NLM history window per neuron |
| `n_synch_out` | 64 | neurons for output-sync |
| `n_synch_action` | 64 | neurons for action-sync |
| `neuron_select_type` | `"first-last"` | first 64 neurons for out, last 64 for action |
| `sequence_length` | 32 | parity task input length |
| `iterations` | 20 | T — thinking steps per forward pass |
| `synch_rep_size` | **2080** | derived: 64×65/2 (upper triangle of 64×64 outer product) |
| `out_dims` | 64 | derived: seq_len × 2 |
| **Parameters** | **1,280,169** | |

```
Baseline: 12.1 steps/sec  |  82.9 ms/step  |  0.10 M tok/s
```

---

## Phase 1 — Framework Optimisations (B=256)

These change nothing about the model architecture. They're pure training-loop improvements applied incrementally.

### 1a. TF32 matrix multiply

```python
torch.set_float32_matmul_precision("high")
```

On Ampere and newer GPUs, float32 matmuls can use tensor cores by rounding inputs to the TF32 format (10-bit mantissa instead of 23-bit). The precision loss is negligible for deep learning. This is a single line set once at program start.

**Result: negligible alone.** The baseline is so latency-bound (kernel launch overhead dominates) that compute precision doesn't matter yet. It becomes free speedup once `torch.compile` takes over.

### 1b. `torch.compile(mode="reduce-overhead")`

```python
model = torch.compile(model, mode="reduce-overhead")
```

This is the dominant optimisation. PyTorch traces the forward pass and compiles it into a sequence of large fused CUDA kernels via the Triton and CUTLASS backends. Instead of launching ~200 separate kernels per training step (one per op in the T=20 recurrent loop), it produces a handful of large kernels that do the same work with far less launch overhead. The `reduce-overhead` mode specifically targets minimising CPU-side kernel scheduling cost using CUDA graphs.

**Result: 5.7× speedup** — the largest single gain of the entire session.

### 1c. bfloat16 autocast

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    ...
```

Runs the forward and backward in bfloat16 (16-bit float with the same exponent range as float32 but 7-bit mantissa). The main benefit is memory bandwidth: all the large weight matrices and activations are half the size, so the GPU's HBM interface is under half the pressure. The NLM einsum (`BNM,MON→BNO`), synapse linear (`Linear(512, 512)`), and attention are all bandwidth-bound at this model size, so bfloat16 gives a real speedup.

bfloat16 is preferred over float16 because it doesn't require a `GradScaler` — its exponent range matches float32, so gradients don't underflow.

**Result: +15% on top of compile** (6.6× total over baseline).

### 1d. Fused AdamW

```python
opt = torch.optim.AdamW(model.parameters(), fused=True)
```

The standard AdamW implementation runs parameter updates one tensor at a time on the CPU's scheduling thread. The fused variant combines all parameter tensors into a single GPU kernel call, eliminating both the per-tensor kernel launch overhead and the CPU-side loop. This is particularly impactful here because the model's gradient update happens after a long recurrent backward pass — the GPU is briefly idle while the CPU loops over parameters.

**Result: +18% on top of compile+bf16** (7.8× total over baseline).

### Phase 1 summary table

| Config | sps | ms/step | M tok/s | Speedup |
|---|---|---|---|---|
| Eager baseline | 12.1 | 82.9 | 0.10 | 1.0× |
| + TF32 | 12.1 | 82.9 | 0.10 | 1.0× |
| + `torch.compile` | 70 | 14.3 | 0.57 | 5.7× |
| + bfloat16 | 80 | 12.6 | 0.65 | 6.6× |
| + fused AdamW | 94 | 10.6 | 0.77 | **7.8×** |

---

## Phase 2 — Profiling the Compiled Model

Before optimising further, `torch.profiler` was run on the compiled+bf16 model (20 steps, sorted by CUDA time):

```
flash_bwd_dq_dk_dv_loop_seqk_par  32.7 ms  (17.9%)   400 calls = 20 iter × T=20
cutlass_80_tensorop_bf16_s16816    23.2 ms  (12.7%)   800 calls
flash_fwd_kernel                   13.1 ms   (7.2%)   400 calls
triton_per_fused_div_index_...      2.7 ms   (1.5%)   400 calls  ← pairwise sync
```

**Total CUDA: 182 ms for 20 steps = 9.1 ms/step**  
**Wall clock: 10.8 ms/step** (1.7 ms CPU overhead remaining)

Key findings:

1. **Flash Attention backward dominates** (32.7 ms, 18%). The attention is called T=20 times per forward pass; its backward (dQ, dK, dV) runs 20 times too. At seq_len=32, Flash Attention is overkill — the sequence is so short that its tiling overhead exceeds the work.

2. **K/V projections recomputed every iteration.** Inside `nn.MultiheadAttention`, the K and V matrices are projected from the input at every one of the T=20 thinking steps. But the input (`kv`) doesn't change across the loop — it's encoded once from the static input sequence. Those 19 extra projections are pure waste.

3. **Pairwise sync allocates a large intermediate.** The `compute_sync` function builds a `(B, 64, 64)` outer product tensor then indexes the upper triangle. At B=256, that's ~4 MB allocated and discarded per call, 2×T=40 times per step.

4. **Circular buffer has no eager benefit.** The `state_trace` update (`torch.cat([trace[:,:,1:], state], dim=-1)`) was expected to be a bottleneck, but GPU memory allocation is fast enough that at B=256 the Python overhead of a pointer-based buffer exceeded the savings.

---

## Phase 3 — Model-Level Optimisations (DeepOptCTM)

`DeepOptCTM` (defined in `bench2.py`) addresses the three bottlenecks found by the profiler. It subclasses `NanoCTM` and overrides `__init__` and `forward`.

### 3a. K/V projection hoisting

The most impactful change. In the baseline, `nn.MultiheadAttention` internally computes:
```
K = W_k @ kv      # (B, seq, d_input) → (B, seq, d_input)
V = W_v @ kv      # same
```
...at every one of the T=20 thinking iterations. Since `kv` is constant (encoded once from the static input), these are identical across all iterations.

`DeepOptCTM` replaces `nn.MultiheadAttention` with separate projection layers and computes K and V **once before the loop**:

```python
# Before the T loop — runs once:
k = self.attn_k(kv).view(B, -1, H, hd).transpose(1, 2)  # (B, H, seq, hd)
v = self.attn_v(kv).view(B, -1, H, hd).transpose(1, 2)

# Inside the T loop — only Q changes each step:
q = self.q_proj(sync_a).view(B, 1, H, hd).transpose(1, 2)
attn_out = F.scaled_dot_product_attention(q, k, v)
```

This saves 19/20 of all K and V projection compute — both forward and backward. The effect scales with batch size: at B=256 the matmuls are too small to matter, but at B=2048 each K/V projection is a `(2048, 32, 256) × (256, 256)` matmul (~860 MFLOP), and running it 20 times vs once is the difference between 17 GFLOP and 860 MFLOP per step.

### 3b. Manual bmm attention

Flash Attention is designed to avoid materialising the full `(B, H, seq, seq)` attention matrix — essential at seq_len=1024+. At seq_len=32, the attention matrix is `(B, H, 1, 32)` — tiny, and Flash Attention's tiling setup dominates the actual computation. Replacing it with two explicit `torch.bmm` calls removes this overhead:

```python
scale  = hd ** -0.5
kt     = k.reshape(B*H, -1, hd).transpose(1, 2)   # (B*H, hd, seq)
scores = torch.bmm(q.view(B*H, 1, hd), kt) * scale  # (B*H, 1, seq)
attn   = scores.softmax(dim=-1)
out    = torch.bmm(attn, v.reshape(B*H, -1, hd))   # (B*H, 1, hd)
```

With `torch.compile`, these two bmms fuse with the surrounding softmax into a single Triton kernel.

### 3c. Direct pairwise sync

The original `compute_sync` builds the full `(B, n_synch, n_synch)` outer product and then extracts the upper triangle:

```python
outer    = sel_l.unsqueeze(2) * sel_r.unsqueeze(1)  # (B, 64, 64) = 1M floats
pairwise = outer[:, i, j]                            # (B, 2080) — upper triangle
```

`DeepOptCTM` pre-computes the combined indices at init time (`idx_left[triu_i]`, `idx_right[triu_j]`) and reads directly from the neuron state:

```python
pairwise = state[:, il_combined] * state[:, ir_combined]  # (B, 2080) direct
```

The `(B, 64, 64)` intermediate allocation vanishes. This is called 2×T=40 times per forward pass.

### 3d. Pre-initialised decay accumulators

The original sync function uses `None` to detect the first call:

```python
if decay_alpha is None:
    decay_alpha = pairwise
    decay_beta  = torch.ones_like(pairwise)
```

This Python-level `None` check creates a conditional branch in the compiled graph. Pre-initialising both accumulators to zero before the loop is numerically identical (`r * 0 + pairwise = pairwise`, `r * 0 + 1 = 1`) but removes the conditional entirely, giving the compiler a single clean code path through the loop.

### Phase 3 results at B=256

At B=256 the individual matmuls are too small to be compute-bound — the improvements above are all below measurement noise:

| Config | sps | M tok/s | Notes |
|---|---|---|---|
| NanoCTM + compile + bf16 + fused | ~100 | 0.82 | ≈ same |
| DeepOptCTM + compile + bf16 + fused | ~100 | 0.82 | ≈ same |

**At B=256 all configs converge to ~100 sps.** The GPU is latency-bound — each matmul is too small to fill 90 SMs, and kernel launch overhead dominates. Architecture changes only show up when the batch is large enough to make the matmuls compute-bound.

---

## Phase 4 — Batch Size Sweep

Finding the GPU saturation point. All configs use DeepOptCTM + compile + bf16 + fused AdamW.

| Batch | sps | ms/step | K samp/s | M tok/s |
|---|---|---|---|---|
| 256 | 98.5 | 10.2 | 25.2 | 0.81 |
| 512 | 88.6 | 11.3 | 45.4 | 1.45 |
| 1024 | 90.1 | 11.1 | 92.3 | 2.95 |
| **2048** | **61.6** | **16.2** | **126.1** | **4.03** ← peak |
| 4096 | 28.3 | 35.3 | 116.1 | 3.71 |
| 8192 | 13.6 | 73.7 | 111.1 | 3.56 |

**B=2048 is the GPU saturation point.** Above this, activation memory for the T=20 recurrent steps (state_trace at each iteration, activated_state at each step) creates bandwidth pressure that outweighs the compute gains.

At peak (B=2048): approximately **6.2 TFLOP/s effective** against a theoretical 220 TFLOPS — **2.8% FLOP utilisation**. The model is too small for the GPU: even with a large batch, the individual matmuls (`Linear(512, 512)`, `Linear(2080, 256)`) don't fill the SM count.

---

## Phase 5 — Architecture: random-pairing vs first-last

The `neuron_select_type` field controls how the synchronisation vector is computed.

**first-last (default):** The model selects the first 64 neurons for output-sync and the last 64 for action-sync. For each group, it computes the full `(n_synch × n_synch)` outer product of neuron activations and extracts the upper triangle. This gives a synch vector of size `n_synch*(n_synch+1)/2 = 2080`.

**random-pairing:** Instead of the outer product, n_synch explicit pairs of neurons are element-wise multiplied. The synch vector has size `n_synch = 64`.

The consequence for speed is in the projections driven by the synch vector:

| | first-last | random-pairing |
|---|---|---|
| `synch_rep_size` | 2080 | 64 |
| `q_proj` params | `Linear(2080, 256)` = 532K | `Linear(64, 256)` = 16K |
| `output_proj` params | `Linear(2080, 64)` = 133K | `Linear(64, 64)` = 4K |
| Total params | 1,280,169 | 631,017 |

At B=256, the synch projections are small enough that the difference is noise. At B=2048, the `q_proj` backward computes a gradient of shape `(2080, 256)` accumulated T=20 times — substantial compute. With random-pairing it's `(64, 256)` — trivial.

**Results at B=2048:**

| Config | sps | M tok/s |
|---|---|---|
| NanoCTM + first-last | 17 | 1.12 |
| NanoCTM + random-pairing | 20 | 1.29 |
| DeepOpt + first-last | 41 | 2.68 |
| **DeepOpt + random-pairing** | **62** | **4.06** |

The combination is multiplicatively better than either alone: KV hoisting removes the K/V projection overhead, and random-pairing removes the q_proj/output_proj overhead. At B=2048 these are both meaningful matmul costs.

---

## Final Results — All Optimisations Stacked

Measuring throughput as M tok/s (sps × B × seq_len / 1e6) to fairly compare across batch sizes.

| Config | sps | ms/step | M tok/s | Speedup |
|---|---|---|---|---|
| Eager baseline (B=256) | 12.1 | 82.9 | 0.10 | 1× |
| + `torch.compile` | 70 | 14.4 | 0.57 | 5.7× |
| + bfloat16 | 80 | 12.6 | 0.65 | 6.6× |
| + fused AdamW | 94 | 10.6 | 0.77 | 7.8× |
| + B=2048 | 17 | 58 | 1.12 | 11× |
| + KV hoisting (DeepOptCTM) | 41 | 24 | 2.68 | 27× |
| + random-pairing config | **62** | **16** | **4.06** | **41×** |

**41× improvement in training throughput** over the out-of-the-box implementation.

---

## What Didn't Help

These were tested and found to have no meaningful impact:

**Circular buffer for state_trace (eager mode).** The baseline uses `torch.cat([trace[:,:,1:], state.unsqueeze(-1)], dim=-1)` to shift the history buffer, allocating a new `(B, d_model, memory_length)` tensor every thinking step. In theory this is T=20 allocations per forward pass. In practice, GPU memory allocation is fast enough that the Python pointer-arithmetic overhead of the circular buffer exceeded the savings. The circular buffer *does* help `torch.compile` produce a cleaner graph (no control-flow dependency on the allocation), so it's kept in DeepOptCTM.

**max-autotune compile mode.** Tried `torch.compile(mode="max-autotune")` which runs an exhaustive search over CUTLASS and Triton kernel configurations. Result: **slower** than `reduce-overhead`. The model's matmuls are too small for the autotuned kernels (which are optimised for throughput at large sizes) — the setup overhead of the autotuned kernels exceeded any kernel-selection gain.

**Removing gradient clipping.** `nn.utils.clip_grad_norm_` was suspected to be a CPU-GPU sync point after the backward pass. Removing it showed no measurable speedup. With fused AdamW, the GPU stream synchronisation happens naturally before the optimizer step regardless, so the norm computation is hidden in the latency.

**Larger batch (B=4096+).** Past B=2048, activation memory for the T=20 steps (all intermediate `state_trace` and `activated_state` values must be retained for backward) creates enough HBM bandwidth pressure to reduce overall throughput.

**TF32 alone.** Without `torch.compile`, TF32 does nothing — the model is kernel-launch-overhead bound, not compute bound. Once compiled, TF32 is already the default via `set_float32_matmul_precision("high")` and contributes to the compile speedup implicitly.

---

## Architecture Note: Why B=256 Shows No Differentiation

At B=256 with `torch.compile`, all model variants produce ~100 steps/sec regardless of architectural differences. This is because the model is **latency-bound**, not compute-bound.

At this model size and batch, the GPU is at approximately 3% of its theoretical FLOP utilisation. The SMs are mostly idle between kernel launches. Any architectural change that reduces FLOPs (KV hoisting, random-pairing) produces matmuls that are just as small relative to the GPU's capacity — the bottleneck is the kernel scheduling latency, not the work done.

The architectural optimisations only matter at B=2048+ where the matmul sizes grow large enough to actually fill some SMs. This is the general principle: **profiling at the wrong batch size gives misleading signals about which optimisations matter**.

---

## Fastest Configuration

```python
torch.set_float32_matmul_precision("high")

cfg = CTMConfig(
    d_model=256, d_input=256, d_embedding=256,
    sequence_length=32, iterations=20, memory_length=20,
    n_synch_out=64, n_synch_action=64,
    neuron_select_type="random-pairing",   # synch_rep 2080 → 64
)

model = torch.compile(DeepOptCTM(cfg).to("cuda"), mode="reduce-overhead")
opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, fused=True)
# batch_size=2048, torch.autocast("cuda", dtype=torch.bfloat16)
```

**4.06 M tok/s — 41× over the eager baseline.**

---

## Files

| File | Purpose |
|---|---|
| `nano_ctm.py` | Model definition + `train(fast=True)` applies compile/bf16/fused/B=2048 |
| `bench.py` | Phase 1 sweep: circular buffer, compile, bf16, TF32 |
| `bench2.py` | Full benchmark: DeepOptCTM, batch sweep, random-pairing, `python bench2.py` |
| `PLAN.md` | Original roadmap; Phases 1–5 addressed here |
| `REPORT.md` | This file |

---

## What's Next

The parity task at seq_len=32 is a sanity check, not a real workload:
- At seq_len=32 the model can solve parity with a single attention pass — the T=20 thinking steps aren't needed
- Certainty jumps from t=0 to high in step 1 or 2; the CTM's adaptive-compute signature is invisible
- All instances are roughly equal difficulty; there's no harder/easier split to reveal differential thinking

The next phase (from `PLAN.md`) is scaling to a task that actually requires thinking:

- **Parity at seq_len=256+**: same task, 8× output space, high variance in instance difficulty (3 sign changes vs 180 sign changes in a 256-token sequence). The model must work across T steps to get all 256 position labels right simultaneously.
- **Sorting**: predict the rank of each element in a random sequence. Requires global pairwise comparison — genuinely impossible in a single step at seq_len=256.

Both tasks require re-evaluating the model size. At d_model=256 the GPU is at 3% utilisation even at B=2048. A meaningful experiment needs d_model ≥ 512 and seq_len ≥ 128 to produce training dynamics worth studying.
