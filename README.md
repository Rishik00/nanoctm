# nanoCTM

nanoGPT but the model is allowed to think before answering. controversial.

A minimal, single-file implementation of the [Continuous Thought Machine](https://sakana.ai/ctm/) (CTM) — cleaned up, commented, and trainable from scratch. Built on the parity task as a sanity check, with GPU optimisations incoming.

## what's in the box

```
nano_ctm.py   — the whole model. one file. read it top to bottom.
PLAN.md       — what's done, what's next, what we're optimising and why.
```

## the idea in one paragraph

The CTM runs your input through T recurrent "thinking" iterations before committing to an answer. Each neuron has its own private linear model over its own history (the NLM). Pairwise neuron correlations (synchronisation) drive attention queries. Predictions are weighted by how confident the model is at each step. It's weird. It works.

## run it

```bash
pip install torch numpy
python nano_ctm.py
```

Logs loss + certainty at t=0 vs t=T. If the model is learning, certainty should grow across thinking steps over the course of training.

## status

- [x] clean baseline
- [ ] training validation
- [ ] circular buffer, `torch.compile`
- [ ] FlashAttention (called T times per step — this one matters)
- [ ] target GPU benchmarks

## reference

> *Continuous Thought Machines* — Sakana AI
