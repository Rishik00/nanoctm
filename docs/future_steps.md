# nanoCTM — Future Directions & Experiments

## Suggested ordering

1. **(5) Documentation schema** — paying this debt now means every experiment from here is automatically recorded.
2. **(2) LightCTM ablation** — directly continues existing optimisation work, gives a faster platform for everything else.
3. **(6a) Dead neuron tracking** — add 5 lines of logging to the training loop, yields a finding regardless of outcome.
4. **(4) Harder task + Phase 4 NLM opt** — associative recall + bmm NLM + gradient checkpointing, properly stresses the GPU.
5. **(3) CTM-LM (no backbone)** — most novel; block-recurrent approach, PTB-char as first target.
6. **(1) CTM-as-FFN in GPT-2** — after (3) is working, the causal machinery already exists.
7. **(6b) GDN / mech interp** — once a harder task is stable and the architecture is locked.

---

## 1. CTM as memory component for GPT-2

The right instinct, but the design question matters a lot. Two concrete variants:

**A. CTM-as-FFN replacement.** Swap each GPT-2 FFN block (2-layer MLP) with a mini-CTM running T=5 iterations. The token hidden state becomes the cross-attention KV; the CTM's recurrent state is "working memory" for that position. This is a local per-token reasoner.

**B. CTM-as-top-level reasoner.** Full GPT-2 backbone → pool → CTM thinks for T steps → decode. Basically the DistilBERT run but generative. The CTM sees the whole context via cross-attention and iterates before emitting a token.

Blocking issue for both: the current architecture is **not causal**. The backbone attends to all positions simultaneously. For LM you need either masked cross-attention in the CTM loop, or confine CTM to a pooled prefix representation. (A) is probably easier to wire up — "recurrent FFN" has precedent in Universal Transformers.

---

## 2. Cutting components to make it lighter and faster

Looking at actual parameter distribution, the fat is concentrated:

- `q_proj` + `output_proj` = **83% of trainable params** in the BERT run (2080-dim sync with first-last). `random-pairing` already cuts synch_rep from 2080→64. Combining with smaller `n_synch_out` (e.g. 32→16) drops these from ~1M params to ~30k.
- The **NLM** (`BNM,MON->BNO`) is the most architecturally unique piece; cutting it entirely makes it a plain RNN. But weight-sharing across neuron groups (e.g. 8 groups of 32 neurons share one NLM) is a clean ablation.
- `memory_length=20` has never been tested at lower values. M=5 or M=10 probably still works for parity; harder tasks will reveal the actual tradeoff.
- The **SynapseNet** is Linear→GLU→LN. Collapsing to a single Linear probably loses nothing on parity. May matter on harder tasks.

Experiment: define a `LightCTM` using random-pairing + n_synch=16 + M=5 + no synapse nonlinearity. Benchmark against full NanoCTM on param count, convergence speed, and final accuracy.

---

## 3. Language modeling with no backbone

The most novel and hardest direction. Core issue: CTM cross-attends to a **static** encoded input. For autoregressive LM, the "input" grows token by token. Two approaches:

**Causal CTM-LM.** Token embeddings as KV, strict lower-triangular mask in the CTM's cross-attention per position. Each CTM invocation generates one token, then that token joins the KV. Inference-time sequential — expensive.

**Block-recurrent CTM-LM.** Process a fixed window (e.g. 128 tokens), run T=20 thinking steps, emit all 128 next-token predictions at once (current architecture already outputs `seq_len × 2`, so `seq_len × vocab_size` is a direct extension). The CTM's recurrent state carries context across blocks — a block-recurrent LM with CTM as the recurrent cell.

The second is more tractable and more interesting — it's close to what Transformer-XL / RWKV / Griffin do, but with CTM dynamics instead of a standard recurrent cell. First target: character-level PTB to keep vocab small.

---

## 4. Faster training on a harder, sizeable task

The profiler finding is the key number: **2.8% FLOP utilisation**. The model is so small that the 5090 is bottlenecked on kernel launch overhead, not compute. To actually stress it, need a model at minimum 50-100× larger.

Unfinished optimisations from the plan (Phase 4):
- **NLM einsum → `torch.bmm`** — likely the biggest remaining kernel-level gain.
- **FlexAttention** — needed for causal masking in LM anyway, solves both problems at once.
- **Gradient checkpointing through the T loop** — activation memory scales with T (not depth), so this is unusually cheap: checkpoint every 4-5 steps, recompute the rest. Enables larger batch at same VRAM.

Target task: **associative recall** (or multi-step arithmetic with carry). These provably require multi-step computation. Parity proved CTM works; the advantage over a single-step model is not yet demonstrated. Associative recall would show it.

---

## 5. Documentation website / experiment changelog

A schema-first approach works best. Define a YAML schema for experiments:

```yaml
# experiments/run3_causal_lm.yaml
id: run3
date: 2026-05-26
intent: "Can CTM do causal LM on PTB without a transformer backbone?"
config:
  d_model: 256
  iterations: 20
  task: ptb-char
hardware: RTX 5090
observations:
  - "Loss stuck at X, gradient through causal mask is sparse"
metrics:
  val_bpc: 1.42
  steps: 5000
result: partial  # success / failure / partial
hypotheses_confirmed: []
next_steps: []
```

A small `render.py` reads all YAMLs and generates a static HTML changelog — no framework, no build step. Alternatively mkdocs-material renders this well and adds free search. The schema is what matters, not the renderer.

---

## 6. Architecture improvements

### High value, doable now

**Dead neuron tracking in NLM.** The NLM is exactly where neuron death would show up — a neuron whose `activated_state` is consistently near zero contributes nothing to sync. Add a hook logging `(activated_state.abs() < 1e-3).float().mean()` per neuron across training. If 20-40% of neurons are dead by step 1000, that's a pruning opportunity and a publishable finding. No published measurements of this on CTMs exist.

**Certainty trajectory mech interp.** Already logging `cert_t0` and `cert_tT`. Next step: log certainty at *every* step and plot trajectory shape per sample. Do hard samples show a different trajectory than easy ones? Does certainty grow monotonically or dip and recover? Zero architecture changes required.

**Deeper SynapseNet.** Currently 1-layer GLU. Adding one more layer (Linear→GLU→Linear→GLU→LN) gives the synapse more capacity to route attention context into state. Cheap ablation.

### Medium effort, high interest

**Sparse attention in CTM.** The seq_len=1024 failure (run1) was gradient collapse through the sync bottleneck + output head scaling — not attention. But sparse attention (local window of ~64 + global CLS token) fixes quadratic cost at long sequences and is a prerequisite for CTM-LM anyway. Use FlexAttention block masks.

**Grouped NLM (partial weight sharing).** Instead of N=256 fully independent NLMs, use 8 groups of 32 neurons sharing one NLM weight matrix. Direct ablation testing how much per-neuron individuality actually matters for CTM dynamics.

### Long-term, novel

**GDN layers as sync replacement.** GDN (Generalized Divisive Normalization, Balle 2015/2016) normalizes each unit by a weighted sum of neighbors' squared activities — conceptually close to what CTM sync does (pairwise neuron correlations). Replacing the sync→projection→attention-query pipeline with a GDN layer is a substantial architectural change. Interesting, but needs a harder task baseline first to measure against.

**Mech interp on learning dynamics.** The open question: what does the sync vector encode over T iterations? Does it specialise (different neurons track different input features)? Are there phase transitions in certainty? Are there "computation heads" analogous to attention heads? No published mech interp work exists on CTMs. This would be genuinely novel, but requires a harder task where interesting dynamics can plausibly emerge.
