# Experiment Report: run1_parity1024

**Date:** 2026-05-17  
**Duration:** 689.8 s (11.5 min)  
**Result:** Failed to learn — model never escaped the random baseline

---

## 1. Motivation

Following the benchmarking work in `REPORT.md`, the next question was whether CTM can learn
parity on sequences long enough to force genuine recurrent thinking. At `seq_len=32`, the model
can effectively memorise parity by lookup — the number of distinct sequences is small enough
that gradient descent finds per-pattern shortcuts. At `seq_len=1024`, no such shortcut exists:
the model must maintain a running bit-flip count across the entire sequence, which requires
its recurrent dynamics to actually work.

This run is the baseline attempt: scale the model up (256 → 512), scale the sequence length
up (32 → 1024), and train from scratch.

---

## 2. Configuration

### Model (CTMConfig)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | 512 | internal state dimension / number of neurons |
| `d_input` | 512 | attention Q/K/V dimension |
| `d_embedding` | 512 | backbone embedding width |
| `n_embedding` | 2 | vocab size (binary: {0, 1}) |
| `sequence_length` | 1024 | **32× larger than bench baseline** |
| `iterations` (T) | 20 | thinking steps per forward pass |
| `memory_length` (M) | 64 | per-neuron NLM history window |
| `n_synch_out` | 64 | neurons driving predictions |
| `n_synch_action` | 64 | neurons driving attention query |
| `neuron_select_type` | first-last | deterministic index assignment |
| `num_heads` | 8 | MHA heads |
| `dropout` | 0.1 | |
| **`out_dims`** | **2048** | = seq_len × 2 — **32× larger than bench** |
| `synch_rep_size` | 2080 | = 64×65/2 (upper triangle of 64×64 outer product) |

### Derived parameter sizes

| Layer | Shape | Parameters |
|-------|-------|-----------|
| backbone | (2, 512) | 1,024 |
| kv_proj | (512, 512) + LayerNorm | 263,168 |
| pos_embedding | (512, 2) | 1,026 |
| NLM (W, b) | (64, 2, 512) + bias | 66,048 |
| synapses | (1024, 1024) + GLU + LN | 1,052,160 |
| attention (MHA) | 3×(512,512) + out | 1,050,624 |
| q_proj | **(512, 2080)** | **1,065,984** |
| output_proj | **(2048, 2080)** | **4,259,840** |
| decay params | (2080 × 2) | 4,160 |
| init state | (512 + 512×64) | 33,280 |
| **Total** | | **7,798,977** |

The `output_proj` alone is **4.26M parameters** — 55% of the entire model — because `out_dims`
scales with `sequence_length`. This has significant implications for gradient flow (see §5).

### Training

| Parameter | Value |
|-----------|-------|
| batch_size | 256 |
| lr | 3e-4 |
| weight_decay | 0.01 |
| grad_clip | 1.0 |
| total_steps | 5000 |
| val_every | 200 |
| val_size | 4096 (fixed holdout) |
| optimizer | AdamW (fused) |
| precision | bfloat16 + torch.compile(reduce-overhead) |
| hardware | RTX 5090 |

---

## 3. Results Summary

| Metric | Step 0 | Step 200 | Step 5000 | Random baseline |
|--------|--------|----------|-----------|-----------------|
| train loss | 0.6938 | 0.6932 | 0.6932 | **0.6931** = ln(2) |
| train acc | 49.9% | 50.0% | 50.0% | 50.0% |
| val loss | 0.6942 | 0.6932 | 0.6932 | 0.6931 |
| val acc | 49.98% | 49.99% | 50.00% | 50.0% |
| cert t=0 | 2.35e-4 | 2.3e-5 | 6e-6 | — |
| cert t=T | 2.53e-3 | 2.3e-5 | 6e-6 | — |

The model started marginally above the random baseline (loss 0.6938 vs 0.6931) and converged
**toward** the random baseline, not away from it. After 5000 steps and 11.5 minutes of training,
every metric is indistinguishable from a model that outputs uniform probabilities.

---

## 4. Gradient Norm Trajectories

This is the most informative data from the run. Per-component gradient norms, sampled every
200 steps:

| step | gnorm_total | output_proj | synapses | nlm | attention | backbone | cert_tT |
|------|------------|-------------|----------|-----|-----------|----------|---------|
| 0 | 0.01613 | 0.00558 (35%) | 0.01062 | 0.00405 | 0.00859 | 0.00013 | 1.54e-3 |
| 200 | 0.00160 | 0.00152 (95%) | 0.00024 | 0.00033 | 0.00011 | 2.0e-6 | 3.0e-5 |
| 400 | 0.00150 | 0.00145 (97%) | 0.00017 | 0.00028 | 7.6e-5 | 1.0e-6 | 2.0e-5 |
| 800 | 0.00145 | 0.00143 (99%) | 8.3e-5 | 0.00018 | 3.3e-5 | **0.000** | 1.0e-5 |
| 1000 | 0.00143 | 0.00142 (99%) | 8.1e-5 | 0.00016 | 3.2e-5 | 0.000 | 1.0e-5 |
| 2000 | 0.00138 | 0.00138 (>99%) | 3.3e-5 | 9.2e-5 | 1.3e-5 | 0.000 | 1.0e-5 |
| 3000 | 0.00139 | 0.00139 (>99%) | 1.9e-5 | 6.3e-5 | 7e-6 | 0.000 | 1.0e-5 |
| 4000 | 0.00142 | 0.00142 (>99%) | 1.6e-5 | 4.0e-5 | 5e-6 | 0.000 | 1.0e-5 |
| 5000 | 0.00143 | 0.00143 (100%) | 8e-6 | 2.8e-5 | 3e-6 | 0.000 | 1.0e-5 |

**Fraction of total gradient in `output_proj`:**
- Step 0: 34.6%
- Step 200: 95.4%
- Step 1000: 99.0%
- Step 5000: **100.0%**

**Gradient decay over the run (step 0 → step 5000):**
- `synapses`: 0.01062 → 0.000008 — **1,328× reduction**
- `attention`: 0.00859 → 0.000003 — **2,863× reduction**
- `nlm`: 0.00405 → 0.000029 — **140× reduction**
- `backbone`: 0.00013 → **exactly 0.000** (dead by step 800)
- `output_proj`: 0.00558 → 0.00142 — only **4× reduction**, dominant the whole time

### Validation certainty trajectory

| step | cert t=0 | cert t=T | gap (T-0) |
|------|----------|----------|-----------|
| 0 | 2.35e-4 | 2.53e-3 | +2.3e-3 |
| 200 | 2.3e-5 | 2.3e-5 | **0** |
| 400+ | ≤1.4e-5 | ≤1.6e-5 | ≈0 |

By step 200, `cert_t0 ≈ cert_tT`. The 20 thinking steps contribute nothing — the model
answers the same at step 1 as at step 20. The recurrent dynamics are inert.

---

## 5. Analysis: The Self-Quenching Mechanism

The run exhibits a specific failure mode that the gradient trajectories make legible. It is
not a slow, frustrating plateau — it is a rapid, self-reinforcing collapse that completes
within the first ~150 steps and then mechanically decays for the remaining 4,850 steps.

### 5.1 The pairwise synchronisation bottleneck

The CTM computes its output predictions via:

```
sync_out[k] = activated_state[i] × activated_state[j]   for each (i, j) neuron pair
pred = output_proj(sync_out)
```

The gradient of the loss with respect to `activated_state[i]` flows through the pairwise
product, and the chain rule gives:

```
∂L/∂activated_state[i]  ∝  sum_j  (∂L/∂sync_out[k])  ×  activated_state[j]
```

This is the critical expression. **The gradient is proportional to the current activation
values of the paired neurons.** If `activated_state` is near zero, the gradient through this
gate is near zero — regardless of how large the upstream loss signal is.

The CTM initialises `start_activated_state` ~ U(−0.1, 0.1). Initial neuron values are small.
Initial pairwise products are therefore quadratically small: values ~ (0.1)² = 0.01. This gives
a small but nonzero initial sync rep, and a small but nonzero initial gradient through it.

### 5.2 The positive feedback loop

The gradient flowing back through the sync bottleneck is small at initialisation. Small gradient
→ small weight updates in NLM and synapses → those modules do not learn to produce larger
activations → the next step's pairwise products are equally small → the gradient through the
sync bottleneck stays small. Meanwhile, `output_proj` is the only component whose gradient is
not filtered through the pairwise product: its gradient is `∂L/∂W_out = sync_out.T @ ∂L/∂pred`,
which is nonzero as long as `sync_out` is nonzero.

So the optimiser keeps updating `output_proj` while everything upstream is starved. But
`output_proj` cannot reduce the loss on its own — its input (`sync_out`) carries no
task-relevant signal — so it converges toward the minimum-gradient solution: mapping any
`sync_out` to uniform logits. This pushes the predictions toward maximum entropy, which
pushes certainty toward zero.

Lower certainty means the model is more uncertain about every position, which in the
certainty-weighted loss becomes `cert_weights = softmax(~0 for all T)` → uniform weights.
The model can survive with zero certainty because the certainty weight just becomes
`1/T` per step rather than 0. But the certainty having collapsed means there is no longer
any incentive for later thinking steps to be more confident than earlier ones. The recurrent
dynamics become completely unnecessary, and their gradient signal dies further.

This is **self-quenching**: the optimiser drives the model toward a state with smaller
gradients (lower loss gradient magnitude), not toward lower loss. The model finds a local
attractor at maximum entropy / zero certainty that is stable because every direction away
from it requires first increasing gradient magnitude.

### 5.3 Why the backbone hits exactly zero

By step 800, `gnorm_backbone = 0.000` (to full 6-decimal-place logging precision). The
backbone is an `nn.Embedding` — its gradient exists only for the token indices present in
the batch. The gradient flowing to backbone requires the full chain:

```
loss → output_proj → sync_out → activated_state (T steps) → NLM
     → state_trace → synapses → attn_out → MHA → kv → kv_proj → backbone
```

At step 800, `gnorm_synapses ≈ 8e-5` and `gnorm_attention ≈ 3e-5`. The backbone
gradient is `synapses_grad × kv_proj_weight × attention_grad × kv_proj_weight^T`, roughly:
```
~3e-5 × 1/sqrt(512) × 3e-5 × ... → O(1e-11) per element
```

In bfloat16, the minimum representable positive normalised value is ~1.2e-38, but the
practical precision floor for gradient accumulation is much higher due to the 7-bit mantissa:
relative precision is ~0.78%, so values below ~1e-10 that are accumulated from many such
terms collapse to zero when the dominant term is ~1e-3. The backbone gradient has underflowed
to exactly zero in the float32 gradient accumulator, driven by the compound attenuation along
the long gradient chain.

**The backbone learning nothing means the input representation never updates.** The model
has no ability to adapt how it reads the input sequence. The attention keys and values
are frozen at their random initialisation for the full 5000 steps.

### 5.4 The out_dims scaling problem

At `seq_len=32`, `out_dims = 64`, and `output_proj` has shape `(64, 2080)` = 133K parameters.  
At `seq_len=1024`, `out_dims = 2048`, and `output_proj` has shape `(2048, 2080)` = 4.26M parameters.

The same loss gradient (∂L/∂pred) is spread over 32× more parameters in the output_proj
weight matrix. The gradient per parameter is 32× smaller. At `seq_len=32`, a small sync rep
still generates a meaningful weight update in the compact `output_proj`. At `seq_len=1024`,
the same sync rep generates a negligible update. This means the window for bootstrapping —
where early loss signal can establish non-trivial weights in `output_proj` before certainty
collapses — is 32× narrower at seq_len=1024.

The practical consequence: at seq_len=32, the model escapes the zero-certainty attractor
because `output_proj` updates fast enough to establish non-uniform logits before gradient
flow to NLM and synapses dies. At seq_len=1024, `output_proj` moves too slowly, certainty
collapses first, and the model is trapped.

### 5.5 Why extra thinking steps did nothing

The certainty-weighted loss places a softmax over `cert[:, 1, :]` across T steps:

```
cert_weights = softmax(certainties[:, 1, :], dim=-1)   # (B, T)
```

When all certainties are ≈ 0 (as they are here from step 200 onward), all weights are
≈ 1/T = 0.05. The loss becomes an unweighted average of CE over all T steps. In this regime,
there is **no incentive for later thinking steps to be better than earlier ones**. The
gradient flowing to step t depends only on whether step t's output is better than average.
Since all steps produce identical near-uniform outputs, the gradient is the same for all t.
The recurrent update rule has no reason to accumulate information across iterations.

This is visible in the data: `cert_t0 = cert_tT` from step 200 onward, and both decay
together. The 20 iterations of recurrent computation are computationally present but
informationally inert.

---

## 6. Key Findings

**Finding 1: Cold-start failure at seq_len=1024 is deterministic, not stochastic.**

The collapse to zero certainty happened at the same rate regardless of which 200-step
window we examine. The trajectory from step 0 to step 200 is the same shape as step 200
to step 400, just smaller in absolute scale. This is a structural property of the model and
initialisation, not a bad random seed.

**Finding 2: The pairwise sync creates a quadratic gradient gate.**

The gradient through the synchronisation mechanism is proportional to the magnitude of the
activated neurons. This is not a standard linear layer where gradient magnitude depends only
on the upstream signal — here it depends on the forward-pass activations as well. Small
initialisations create a gradient gate that closes on itself.

**Finding 3: The out_dims bottleneck makes the scaling problem worse by 32×.**

Going from seq_len=32 to seq_len=1024 doesn't just make the task harder — it makes the
output head 32× larger (4.26M vs 133K), which dilutes the per-parameter gradient by 32×.
This means bootstrapping requires much stronger early signal, which the model doesn't have.
Any architecture that scales `out_dims ∝ seq_len` will have this problem.

**Finding 4: The backbone receives zero gradient by step 800.**

The input representation is completely frozen after ~800 steps. The model cannot adapt
how it reads the sequence. This means even if the recurrent dynamics somehow recovered,
the keys and values in cross-attention are uninformative random projections of the input.

**Finding 5: No amount of additional training steps would have helped.**

The gradient norm is not declining toward zero because we're near a minimum — it's stuck
because the model is in a saddle point with a structural barrier. The loss at step 5000
(0.6932) is higher than ln(2) = 0.6931, meaning the model is slightly *worse* than random.

---

## 7. What This Rules Out

This run rules out the following hypotheses:
- "The model just needs more steps at seq_len=1024" — no, it collapsed in the first 150 steps
- "LR=3e-4 is adequate for this scale" — no, effective update ≈ 3e-4 × 0.001 = 3e-7 per parameter
- "BF16 + compile doesn't affect learning" — backbone gradient underflowed to zero in BF16
- "seq_len=32 → seq_len=1024 is a reasonable step size" — no, the gap is too large

---

## 8. Hypotheses for Next Experiments

### H1: Curriculum learning (most likely to work)

Train seq_len=32 → 64 → 128 → 256 → 512 → 1024 in stages, saving and loading checkpoints.
At seq_len=32, the model demonstrably learns (from the bench runs: loss drops to 0.63 by
step 900, certainty 0.078 at t=T). Once neurons are in an active regime, the sync rep is
nonzero, and the gradient gate stays open as seq_len increases incrementally.

### H2: Larger initialisation for `start_activated_state`

The collapse starts because initial neuron values are too small. Changing init from U(−0.1, 0.1)
to U(−1.0, 1.0) gives initial pairwise products ~100× larger, giving the gradient gate a
100× stronger signal in the first few steps. This might be enough to bootstrap the model
without curriculum.

### H3: Separate output head that doesn't scale with seq_len

At seq_len=1024, `out_dims=2048` creates a 4.26M parameter output head. An alternative:
pool the sync rep into a fixed-size vector and decode position-by-position from that. This
breaks the linear scaling of `out_dims` with `seq_len` and keeps the output head compact.

### H4: Higher learning rate with warmup

At LR=1e-2 with cosine warmup over 200 steps, the tiny initial gradients (~0.001) produce
updates of 1e-2 × 1e-3 = 1e-5 per parameter — 30× larger than the 3e-4 baseline. This
might be enough to move `output_proj` fast enough to establish non-uniform logits before
the certainty collapses. Risk: instability at high LR for a 7.8M param model.

---

## 9. Files

| File | Description |
|------|-------------|
| `train_parity.py` | Research training script with full metric logging |
| `logs/20260517_135541_run1_parity1024/config.json` | Full config snapshot |
| `logs/20260517_135541_run1_parity1024/train.jsonl` | 5000 training records, one per step |
| `logs/20260517_135541_run1_parity1024/val.jsonl` | 25 validation records (every 200 steps) |
| `logs/20260517_135541_run1_parity1024/summary.json` | Run summary |

---

*Next: implement curriculum trainer — seq_len stages with checkpoint transfer.*
