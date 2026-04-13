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

## Phase 1: Training Validation
Confirm the model actually learns before optimising anything.

- [ ] Run full training on parity (`sequence_length=64`, `iterations=5`)
- [ ] Add per-step accuracy to the training log (not just loss)
- [ ] Verify **certainty grows from t=0 → t=T** across training — this is the key
      behavioural signature of CTM working correctly
- [ ] Establish baseline wall-clock time per step (CPU and target GPU)

---

## Phase 2: Easy Wins (no architecture changes)

- [ ] **Circular buffer for `state_trace`**
  - Currently: `torch.cat([trace[:,:,1:], state.unsqueeze(-1)], dim=-1)` — allocates a
    new tensor every iteration × every step
  - Fix: write-index + modular indexing, reorder via `torch.roll` or explicit index gather
  - Expected gain: removes `T × memory_length` allocations per forward pass

- [ ] **`torch.compile` the model**
  - `model = torch.compile(model)` — single line, often 10–30% speedup on GPU
  - Try `mode="reduce-overhead"` first, then `mode="max-autotune"` for the target GPU
  - Watch for graph breaks (the state_trace cat is a likely culprit — fix circular buffer first)

- [ ] **Profile the baseline**
  - Use `torch.profiler` to find the actual bottleneck before guessing
  - Candidates: NLM einsum, sync outer product, state_trace cat, attention

---

## Phase 3: Attention Optimisation
The attention is called **T times per forward pass** — any speedup here multiplies by T.

- [ ] **FlashAttention via `F.scaled_dot_product_attention`**
  - Replace `nn.MultiheadAttention` with manual Q/K/V projections + `F.sdpa`
  - `F.sdpa` uses FlashAttention automatically when inputs are on CUDA + correct dtype
  - Already have `need_weights=False` which is a prerequisite

- [ ] **FlexAttention (PyTorch ≥ 2.5)**
  - Useful if we want custom attention masks or relative bias in future tasks
  - Worth trying on the target GPU to compare vs plain sdpa

- [ ] **Q shape** — current Q is `(B, 1, d_input)` (single query per step)
  - This is already the cheapest possible attention pattern; confirm the GPU
    is still compute-bound here (if not, attention isn't the bottleneck)

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

## Phase 5: Sync Computation
- [ ] The outer product + triu indexing allocates intermediates every step
  - For `n_synch=32`, outer is `(B, 32, 32)` — small, but called `2 × T` times per batch
  - Consider pre-computing the triu index tensors once and reusing (already done for
    `idx_left/right` — same principle for `i, j = triu_indices(...)`)
- [ ] Register `i, j` triu indices as buffers (currently recomputed every forward call)

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

## Notes
- Always profile before optimising — don't guess the bottleneck
- Optimise in order: correctness → profiling → easy wins → targeted fixes
- Keep the baseline `train()` runnable at every stage so regressions are obvious
