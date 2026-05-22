# nanoCTM

nanoGPT but the model is allowed to think before answering. controversial.

A minimal, single-file implementation of the [Continuous Thought Machine](https://sakana.ai/ctm/) (CTM) — cleaned up, trainable from scratch, and progressively optimised. The goal is to understand how CTMs work from the inside: decompose the architecture, figure out where the compute goes, make it fast, and find out what actually breaks when you try to scale them.

## what's in the box

```
nano_ctm.py                    — the whole model. one file. read it top to bottom.
train_parity.py                — research training script with full metric logging
triain_bert_nanoctm.py         — CTM head on frozen DistilBERT, SST-2
PLAN.md                        — roadmap; what's done and what's next
reports/REPORT.md              — full benchmarking report (phases 1–5)
reports/REPORT_RUN1.md         — run1: parity at seq_len=1024, failure analysis
reports/bert_report.md         — run2: CTM head on frozen DistilBERT, SST-2
```

## the idea

The CTM runs your input through T recurrent "thinking" iterations before committing to an answer. Each neuron has its own private linear model over its own history (the NLM). Pairwise neuron correlations (synchronisation) drive attention queries. Predictions are weighted by how confident the model is at each step. The key behavioural signature: certainty should grow from t=0 → t=T as the model "thinks". If it doesn't, the recurrent dynamics are doing nothing.

## run it

```bash
pip install torch numpy
python nano_ctm.py
```

Pass `train(fast=True)` to enable `torch.compile + bfloat16 + fused AdamW + B=2048` — the full optimised stack.

## progress & results

### throughput optimisation (RTX 5090, parity seq_len=32, B=256→2048)

Starting from eager PyTorch float32, each optimisation was stacked and measured:

| Config | sps | ms/step | M tok/s | Speedup |
|---|---|---|---|---|
| Eager baseline (B=256) | 12.1 | 82.9 | 0.10 | 1× |
| + `torch.compile(reduce-overhead)` | 70 | 14.4 | 0.57 | 5.7× |
| + bfloat16 autocast | 80 | 12.6 | 0.65 | 6.6× |
| + fused AdamW | 94 | 10.6 | 0.77 | 7.8× |
| + B=2048 (saturation point) | 17 | 58 | 1.12 | 11× |
| + KV hoisting (`DeepOptCTM`) | 41 | 24 | 2.68 | 27× |
| + random-pairing sync | **62** | **16** | **4.06** | **41×** |

**41× improvement** over the out-of-the-box implementation. Full config:

```python
torch.set_float32_matmul_precision("high")
cfg = CTMConfig(sequence_length=32, iterations=20, neuron_select_type="random-pairing")
model = torch.compile(DeepOptCTM(cfg).cuda(), mode="reduce-overhead")
opt   = torch.optim.AdamW(model.parameters(), fused=True)
# batch_size=2048, torch.autocast("cuda", dtype=torch.bfloat16)
```

### what the profiler found

After `torch.compile + bf16`, `torch.profiler` on the compiled model (20 steps):

- **Flash Attention backward dominates** (18% of CUDA time). It's called T=20 times per forward pass. At seq_len=32 its tiling overhead exceeds the actual work.
- **K/V projections recomputed every iteration.** `kv` (the encoded input) doesn't change across the T loop — those 19/20 extra projections are pure waste.
- **Pairwise sync allocates a (B, 64, 64) intermediate** 2×T=40 times per step (~4 MB per call at B=256).
- **Circular buffer has no eager benefit.** GPU alloc is fast enough that Python pointer arithmetic overhead exceeded savings. Still kept in `DeepOptCTM` for cleaner compiled graph.

`DeepOptCTM` fixes all three: hoists K/V before the loop, replaces Flash Attention with two fused `bmm` calls, and eliminates the outer product intermediate via direct combined-index reads.

### batch size sweep

B=2048 is the GPU saturation point. Above this, activation memory for T=20 steps (all `state_trace` and `activated_state` retained for backward) creates bandwidth pressure.

| Batch | sps | K samp/s | M tok/s |
|---|---|---|---|
| 256 | 98.5 | 25.2 | 0.81 |
| 512 | 88.6 | 45.4 | 1.45 |
| 1024 | 90.1 | 92.3 | 2.95 |
| **2048** | **61.6** | **126.1** | **4.03** ← peak |
| 4096 | 28.3 | 116.1 | 3.71 |

At peak: ~6.2 TFLOP/s effective vs 220 TFLOPS theoretical — **2.8% FLOP utilisation**. The model is too small for the GPU at this task size.

### what didn't help

- `max-autotune` compile mode — **slower** than `reduce-overhead`. Autotuned kernels are optimised for large matmuls; setup overhead exceeded gains at this model size.
- Larger batch (B=4096+) — activation memory for the T=20 recurrent steps kills throughput.
- Removing gradient clipping — no measurable effect; GPU sync happens anyway before fused AdamW.
- TF32 alone — does nothing without `torch.compile` (model is kernel-launch-overhead bound, not compute bound).

### architecture: random-pairing vs first-last

`neuron_select_type` controls the sync computation. `first-last` uses the full `(n_synch × n_synch)` outer product → synch_rep_size=2080. `random-pairing` multiplies n_synch explicit pairs → synch_rep_size=64. This changes the `q_proj` from `Linear(2080, 256)` to `Linear(64, 256)` — 33× fewer params in the most frequently computed projection.

At B=256: no difference (latency bound). At B=2048: the q_proj backward becomes a real cost.

| Config | sps @ B=2048 | M tok/s |
|---|---|---|
| NanoCTM + first-last | 17 | 1.12 |
| NanoCTM + random-pairing | 20 | 1.29 |
| DeepOpt + first-last | 41 | 2.68 |
| **DeepOpt + random-pairing** | **62** | **4.06** |

### run1: parity at seq_len=1024 (failed)

**Date:** 2026-05-17 | **Duration:** 11.5 min | **Outcome:** model never escaped random baseline

Scaled up to d_model=512, seq_len=1024, T=20. Loss stayed at ln(2)=0.6931 for all 5000 steps. Key findings:

**Gradient collapse via pairwise sync bottleneck.** The gradient through the sync mechanism is proportional to the *current* neuron activations — `∂L/∂state[i] ∝ Σⱼ (∂L/∂sync[k]) × state[j]`. Small initialisations (U(-0.1, 0.1)) produce quadratically small pairwise products (~0.01), giving the sync gate a small gradient from the start. Small gradient → no learning → activations stay small → gradient stays small. Self-quenching.

**The out_dims scaling problem amplifies this by 32×.** `output_proj` shape is `(out_dims, 2080)` where `out_dims = seq_len × 2`. Going from seq_len=32 to 1024 makes the output head 4.26M parameters (55% of the whole model), diluting per-parameter gradient by 32×. The bootstrapping window — where early loss signal can establish non-uniform logits before certainty collapses — is 32× narrower.

**Gradient share of `output_proj` over training:**

| step | output_proj share | other components |
|---|---|---|
| 0 | 35% | synapses 66%, attention 53%, nlm 25% |
| 200 | 95% | everything else starved |
| 5000 | **100%** | backbone dead at step 800 |

**Backbone receives exactly zero gradient by step 800.** The input representation is frozen. Attention keys and values are random projections of input for the full 5000 steps.

**Certainty collapses by step 200** and stays flat — `cert_t0 ≈ cert_tT` from that point. The 20 thinking steps are computationally present but informationally inert.

**Next experiments:**
- **H1 (most likely):** Curriculum learning — train seq_len=32→64→128→256→512→1024 with checkpoint transfer. At seq_len=32 the model demonstrably learns; keep neurons in an active regime as seq_len grows.
- **H2:** Larger init for `start_activated_state` — U(-0.1, 0.1) → U(-1.0, 1.0) gives 100× larger initial pairwise products.
- **H3:** Output head that doesn't scale with seq_len — pool sync rep to fixed size, decode position-by-position.

### run2: CTM head on frozen DistilBERT — SST-2 (2026-05-21)

**Setup:** frozen DistilBERT backbone (66M params, untrained), 1.3M trainable CTM head, 2001 steps (~2 epochs), bfloat16, fused AdamW, lr=1e-3, B=64.

**Result: 86.35% val accuracy** on SST-2 binary sentiment classification.

| Model | Val Acc | Trainable params |
|---|---|---|
| NanoCTM + frozen DistilBERT (this run) | **86.35%** | 1.3M (~2%) |
| DistilBERT fine-tuned end-to-end | ~91% | 66M |
| BERT-base fine-tuned end-to-end | ~93% | 110M |

**Certainty dynamics confirmed.** cert_tT consistently leads cert_t0 throughout training (0.001 → ~0.65 by step 1000). The CTM's iterative refinement loop is doing real work — not just learning to be confident, but actually improving across the T=20 steps.

**Training curve:**

| Step | Loss | Acc | cert t=0 | cert t=T |
|---|---|---|---|---|
| 0 | 0.694 | 51.6% | 0.001 | 0.001 |
| 100 | 0.386 | 79.7% | 0.285 | 0.362 |
| 500 | 0.443 | 79.7% | 0.430 | 0.436 |
| 900 | 0.181 | 95.3% | 0.539 | 0.571 |
| 1000 | 0.275 | 87.5% | 0.648 | 0.656 |

**What worked:** frozen backbone gives rich enough representations for the CTM head to orient quickly (~80% in first 100 steps / ~6k samples seen).

**What needs work:**
- Loss is noisy — flat lr=1e-3 is too high for late training; warmup + cosine decay would fix this
- Only ~2 epochs; 3–5 expected to saturate; likely 88–90%+ with same setup and more steps
- `q_proj` and `output_proj` each have 532k params (2080×256, from first-last sync) — 83% of all trainable params in just two layers. Cutting n_synch from 64 to 32 would shrink synch_rep from 2080 → 528 with likely little accuracy loss
- Backbone is frozen — outputs are identical every epoch. Pre-caching them would eliminate the backbone forward pass entirely
- State trace still allocates a new `(B, d_model, memory_length)` tensor at each of T=20 iterations

## reference

> *Continuous Thought Machines* — Sakana AI
