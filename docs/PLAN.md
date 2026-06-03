# nanoCTM — Project Plan

## Phase 0: Baseline ✅
- [x] `nano_ctm.py` — clean single-file implementation
  - `CTMConfig` dataclass (all hyperparams in one place)
  - `NLM`, `SynapseNet`, `RotaryPositionalEmbedding`
  - `NanoCTM` with full recurrent forward pass
  - `ctm_loss` — certainty-weighted CE across T iterations
  - `train()` — baseline training loop with certainty logging
- [x] Forward + backward pass verified (290k params, parity task)

---

## Phase 1: Training Validation ✅
Confirm the model actually learns before optimising anything.

- [x] Run full training on parity (`sequence_length=64`, `iterations=5`)
- [x] Add per-step accuracy to the training log (not just loss)
- [x] Verify **certainty grows from t=0 → t=T** across training — this is the key
      behavioural signature of CTM working correctly
- [x] Establish baseline wall-clock time per step (CPU and target GPU)

---

## Phase 2: Easy Wins (no architecture changes) ✅

- [x] **Circular buffer for `state_trace`**
  - Eager benefit negligible (GPU alloc fast enough), but cleans up compiled graph — kept in DeepOptCTM
- [x] **`torch.compile(mode="reduce-overhead")`** — **5.7× speedup**, dominant single gain
- [x] **bfloat16 autocast** — +15% on top of compile (6.6× total)
- [x] **Fused AdamW** — +18% on top of compile+bf16 (7.8× total)
- [x] **Profile the baseline** — ran `torch.profiler`, found Flash Attention backward (18%), K/V recomputation, and pairwise sync allocation as top bottlenecks

---

## Phase 3: Attention Optimisation ✅
The attention is called **T times per forward pass** — any speedup here multiplies by T.

- [x] **K/V projection hoisting** (biggest win) — K and V are computed from static input; moved
  outside the T loop, saving 19/20 of all K/V projection compute. At B=2048: ~860 MFLOP → 43 MFLOP.
- [x] **Replace `F.sdpa` / Flash Attention with manual bmm** — at seq_len=32, Flash Attention's
  tiling overhead exceeds the work. Two explicit `torch.bmm` calls fuse into a single Triton kernel
  under `torch.compile`.
- [x] **Q shape confirmed** — single query `(B, 1, d_input)` is already the cheapest pattern.
- [ ] **FlexAttention** — not yet tried; relevant if adding custom masks for harder tasks.

---

## Phase 4: NLM / Matmul Optimisation
The NLM einsum `'BNM,MON->BNO'` is a batched matmul over N=d_model neurons.

- [ ] Benchmark: **einsum vs explicit `torch.bmm`**
  - Reshape `W (M, O, N)` → `(N, M, O)`, input `(B, N, M)` → `(B*N, 1, M)`
  - `bmm` may be faster if the einsum decomposition isn't being fused
- [ ] Try `torch.compile` on NLM in isolation (likely to fuse well — pure tensor ops)
- [ ] Consider **half-precision (bfloat16)** for the NLM weights
  - NLM is the most parameter-dense part; bf16 halves memory bandwidth pressure

---

## Phase 5: Sync Computation ✅
- [x] **Direct pairwise indexing** — replaced `(B, 64, 64)` outer product + triu extraction with
  direct element-wise product using pre-computed combined indices. Eliminates the 4 MB intermediate
  allocation called 2×T=40 times per step.
- [x] **Pre-initialised decay accumulators** — removed `None` branch from sync function, giving
  the compiler a single clean code path through the loop.

---

## Phase 6: Target GPU + Task
Once optimisations are in, pick a real benchmark to measure against.

- [ ] **Choose target GPU** — A100 / H100 / whatever is available
- [ ] **Choose a harder task** — parity is a proof-of-concept; options:
  - Associative recall (attention-heavy)
  - Multi-step arithmetic
  - Sequential MNIST
- [ ] **Benchmarking harness**
  - Measure: steps/sec, memory usage, FLOP utilisation (via `torch.profiler`)
  - Compare: baseline → +circular buffer → +compile → +FlashAttn → +bf16
  - Each optimisation should be a measurable, isolated improvement

---

## Phase 7: CTM as a Task Head (NLP)

Using the CTM as a lightweight reasoning head on top of a frozen pretrained backbone.

- [x] **SST-2 binary sentiment (DistilBERT backbone)** — 86.35% val accuracy, 1.3M trainable
  params, ~2 epochs. Certainty dynamics confirmed: cert_tT > cert_t0 throughout training.
  See `reports/bert_report.md`.
- [ ] **LR schedule** — warmup + cosine decay; flat lr=1e-3 causes loss noise in late training
- [ ] **More epochs** — 3–5 expected to saturate; 88–90%+ likely with same setup
- [ ] **Reduce n_synch** — cut 64 → 32, synch_rep 2080 → 528; 83% of trainable params currently
  live in q_proj + output_proj
- [ ] **Pre-cache backbone outputs** — backbone is frozen; re-running it per epoch is pure waste
- [ ] **Harder NLP tasks** — e.g. MultiNLI (3-class), CoLA (linguistic acceptability), tasks
  that plausibly require multi-step reasoning to reveal the CTM's advantage

---

## Notes
- Always profile before optimising — don't guess the bottleneck
- Optimise in order: correctness → profiling → easy wins → targeted fixes
- Keep the baseline `train()` runnable at every stage so regressions are obvious
