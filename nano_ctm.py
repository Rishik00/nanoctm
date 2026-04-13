"""
nano_ctm.py
-----------
A clean, minimal implementation of the Continuous Thought Machine (CTM),
in the spirit of nanoGPT: one file, readable, trainable from scratch.

Architecture in one paragraph:
    The model encodes input into static key/value pairs (via a backbone).
    It then runs T recurrent "thinking" iterations. Each iteration uses
    pairwise neuron correlations (synchronisation) to form an attention query,
    attends over the input KVs, feeds the result through a synapse network to
    produce a new internal state, and updates each neuron's personal history.
    A per-neuron linear model (NLM) reads that history to produce the next
    activated state. A separate out-synchronisation projects the final neuron
    correlations to per-step predictions. Certainty (1 - normalised entropy)
    weights the loss across iterations.

Reference: "Continuous Thought Machine" — Sakana AI
"""

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CTMConfig:
    """All hyperparameters in one place. Change here, nowhere else."""

    # Model dimensions
    d_model: int = 128          # Number of neurons (size of the internal state vector)
    d_input: int = 128          # Dimension used for attention Q/K/V
    d_embedding: int = 128      # Backbone embedding output dimension
    n_embedding: int = 2        # Embedding vocabulary size (parity: {0, 1})

    # Attention
    num_heads: int = 8

    # Memory / NLM
    memory_length: int = 10     # History window length per neuron

    # Synchronisation
    n_synch_out: int = 32       # Neurons used for output-sync (drives predictions)
    n_synch_action: int = 32    # Neurons used for action-sync (drives attention query)
    neuron_select_type: Literal["first-last", "random", "random-pairing"] = "first-last"

    # Task
    sequence_length: int = 64

    # Training
    dropout: float = 0.01
    iterations: int = 5         # T: number of thinking steps per forward pass

    # Derived — set automatically, do not pass manually
    out_dims: int = field(init=False)
    synch_rep_size_action: int = field(init=False)
    synch_rep_size_out: int = field(init=False)

    def __post_init__(self):
        self.out_dims = self.sequence_length * 2
        self.synch_rep_size_action = _sync_rep_size(self.neuron_select_type, self.n_synch_action)
        self.synch_rep_size_out    = _sync_rep_size(self.neuron_select_type, self.n_synch_out)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ParityDataset(Dataset):
    """
    Parity task: given a sequence of ±1 values, predict for each position
    whether the running count of -1s so far is even (0) or odd (1).

    The task requires the model to maintain state across the sequence —
    a simple but meaningful test of recurrent memory.

    Returns:
        vector: (sequence_length,) float32 tensor of {-1.0, +1.0}
        target: (sequence_length,) int64  tensor of {0, 1}
    """

    def __init__(self, sequence_length: int = 64, length: int = 100_000):
        self.sequence_length = sequence_length
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vector = (2 * torch.randint(0, 2, (self.sequence_length,)) - 1).float()
        negatives = (vector == -1).long()
        cumsum = torch.cumsum(negatives, dim=0)
        target = (cumsum % 2 != 0).long()
        return vector, target


# ---------------------------------------------------------------------------
# Positional Embedding
# ---------------------------------------------------------------------------

class RotaryPositionalEmbedding(nn.Module):
    """
    Learned positional embedding based on rotating a unit vector.

    For each position i in [0, seq_len), a unit vector [0, 1] is rotated by
    an angle linearly spaced in [0°, 180°]. The 2D rotated vectors are then
    projected to d_model via a learned linear layer.

    This gives each position a unique, smooth embedding without a lookup table.

    Input shape:  (B, d_model, seq_len)  — only shape is used, not values
    Output shape: (B, d_model, seq_len)  — positional encoding to add to input
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.projection = nn.Linear(2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(2)
        start = torch.tensor([0.0, 1.0], device=x.device)

        # Angles uniformly spaced from 0° to 180° across the sequence
        thetas = torch.deg2rad(torch.linspace(0.0, 180.0, seq_len, device=x.device))
        cos_t = torch.cos(thetas).unsqueeze(1)  # (seq_len, 1)
        sin_t = torch.sin(thetas).unsqueeze(1)

        # 2×2 rotation matrices for each position: (seq_len, 2, 2)
        rot = torch.stack([
            torch.cat([ cos_t, -sin_t], dim=1),
            torch.cat([ sin_t,  cos_t], dim=1),
        ], dim=1)

        rotated = torch.einsum('sij,j->si', rot, start)     # (seq_len, 2)
        pe = self.projection(rotated).transpose(0, 1)       # (d_model, seq_len)
        return pe.unsqueeze(0).expand(x.size(0), -1, -1)   # (B, d_model, seq_len)


# ---------------------------------------------------------------------------
# Neuron-Level Model (NLM)
# ---------------------------------------------------------------------------

class NLM(nn.Module):
    """
    Neuron-Level Model: each of the d_model neurons has its own independent
    linear model that maps its personal history → next activation.

    This is what makes CTM different from a standard RNN. Rather than a shared
    transition function over the full state, each neuron is an autonomous unit
    with its own weights. The full computation is batched via einsum.

    Einsum convention:
        'BNM, MON -> BNO'
         │││   │││
         │││   ││└─ N neurons (one model per neuron, in parallel)
         │││   │└── O output dims (2 before GLU → 1 after)
         │││   └─── M memory steps (history length)
         ││└─ M memory steps
         │└── N neurons
         └─── B batch

    Args:
        memory_length: M — how many past states each neuron can see
        d_model:       N — number of neurons
        out_dims:      O — set to 2 so GLU(dim=-1) halves it to 1
        dropout:       applied to the input trace before layernorm
    """

    def __init__(
        self,
        memory_length: int,
        d_model: int,
        out_dims: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout   = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(memory_length, elementwise_affine=True)

        bound = 1.0 / math.sqrt(memory_length + out_dims)
        self.W = nn.Parameter(
            torch.empty(memory_length, out_dims, d_model).uniform_(-bound, bound)
        )
        self.b = nn.Parameter(torch.zeros(1, d_model, out_dims))
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, state_trace: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_trace: (B, N, M) — each neuron's history over memory_length steps

        Returns:
            activated: (B, N) — new activated state for each neuron
        """
        x = self.dropout(state_trace)
        x = self.layernorm(x)                                   # (B, N, M)
        x = torch.einsum('BNM,MON->BNO', x, self.W) + self.b   # (B, N, 2)
        x = F.glu(x, dim=-1)                                    # (B, N, 1)
        return x.squeeze(-1) / self.temperature                 # (B, N)


# ---------------------------------------------------------------------------
# Synapse Network
# ---------------------------------------------------------------------------

class SynapseNet(nn.Module):
    """
    The synapse network maps the concatenated (attention_output || activated_state)
    to a new pre-NLM internal state.

    Input:  (B, d_input + d_model)
    Output: (B, d_model)
    """

    def __init__(self, d_input: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_input + d_model, d_model * 2),  # × 2 because GLU halves it back to d_model
            nn.GLU(dim=-1),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Synchronisation helpers
# ---------------------------------------------------------------------------

def _sync_rep_size(neuron_select_type: str, n_synch: int) -> int:
    """
    Size of the synchronisation vector produced from n_synch neurons.

    'first-last' / 'random'   → upper triangle of (n_synch × n_synch) outer product
                                 = n_synch * (n_synch + 1) / 2
    'random-pairing'          → element-wise product of explicit pairs
                                 = n_synch
    """
    if neuron_select_type == "random-pairing":
        return n_synch
    return (n_synch * (n_synch + 1)) // 2


def _init_neuron_indices(
    neuron_select_type: str,
    synch_type: str,
    d_model: int,
    n_synch: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (left_indices, right_indices) for synchronisation.

    'first-last': deterministic — first n neurons for 'out', last n for 'action'
    'random' / 'random-pairing': randomly sampled at model init (fixed, not per-step)
    """
    if neuron_select_type == "first-last":
        if synch_type == "out":
            idx = torch.arange(0, n_synch, device=device)
        else:  # action
            idx = torch.arange(d_model - n_synch, d_model, device=device)
        return idx, idx   # left == right for first-last

    elif neuron_select_type in ("random", "random-pairing"):
        left  = torch.from_numpy(np.random.choice(d_model, size=n_synch, replace=False)).to(device)
        right = torch.from_numpy(np.random.choice(d_model, size=n_synch, replace=False)).to(device)
        return left, right

    raise ValueError(f"Unknown neuron_select_type: {neuron_select_type!r}")


def compute_sync(
    activated_state: torch.Tensor,
    decay_alpha: Optional[torch.Tensor],
    decay_beta:  Optional[torch.Tensor],
    r: torch.Tensor,
    neuron_select_type: str,
    n_synch: int,
    idx_left: torch.Tensor,
    idx_right: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the synchronisation vector: an exponentially decayed running average
    of pairwise neuron products, normalised by the square root of the decay
    accumulator (analogous to a running standard deviation normalisation).

    On the first call, pass decay_alpha=None and decay_beta=None.
    On subsequent calls, pass the returned (decay_alpha, decay_beta) back in.

    Args:
        activated_state: (B, d_model)
        decay_alpha:     (B, synch_rep_size) or None
        decay_beta:      (B, synch_rep_size) or None
        r:               (B, synch_rep_size) — per-element decay rates ∈ (0, 1)
        neuron_select_type: controls how pairwise products are computed
        n_synch:         number of neurons selected per side
        idx_left/right:  neuron index tensors (registered buffers in NanoCTM)

    Returns:
        synchronisation: (B, synch_rep_size)
        updated decay_alpha, decay_beta
    """
    if neuron_select_type == "random-pairing":
        # Element-wise product of explicit neuron pairs
        pairwise = activated_state[:, idx_left] * activated_state[:, idx_right]  # (B, n_synch)
    else:
        # Outer product, upper triangle only (matrix is symmetric so no double-counting)
        sel_left  = activated_state[:, idx_left]    # (B, n_synch) — left == right for first-last
        sel_right = activated_state[:, idx_right]
        outer = sel_left.unsqueeze(2) * sel_right.unsqueeze(1)          # (B, n_synch, n_synch)
        i, j  = torch.triu_indices(n_synch, n_synch, device=activated_state.device)
        pairwise = outer[:, i, j]                                        # (B, synch_rep_size)

    # Recurrent decay update — exponential moving average of pairwise products
    # Check both alpha and beta together so type narrowing works correctly.
    if decay_alpha is None or decay_beta is None:
        decay_alpha = pairwise
        decay_beta  = torch.ones_like(pairwise)
    else:
        decay_alpha = r * decay_alpha + pairwise
        decay_beta  = r * decay_beta  + 1.0

    synchronisation = decay_alpha / decay_beta.sqrt()
    return synchronisation, decay_alpha, decay_beta


# ---------------------------------------------------------------------------
# Entropy / certainty helpers
# ---------------------------------------------------------------------------

def _normalised_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute normalised entropy of a softmax distribution: H / H_max ∈ [0, 1].
    When logits.dim() > 2, averages entropy over all non-batch dimensions.

    Args:
        logits: (..., num_classes)

    Returns:
        norm_ent: (B,)
    """
    probs    = logits.softmax(dim=-1)
    log_p    = logits.log_softmax(dim=-1)
    entropy  = -(probs * log_p).sum(dim=-1)
    max_ent  = math.log(logits.size(-1))
    norm_ent = entropy / max_ent
    if logits.dim() > 2:
        norm_ent = norm_ent.flatten(1).mean(-1)
    return norm_ent


def compute_certainty(prediction: torch.Tensor, reshaper: list) -> torch.Tensor:
    """
    Returns (B, 2): [normalised_entropy, 1 - normalised_entropy]
                     i.e. [uncertainty,   certainty]

    Args:
        prediction: (B, out_dims) — flat logit vector
        reshaper:   list to reshape prediction before computing entropy,
                    e.g. [sequence_length, 2] for parity
    """
    B = prediction.size(0)
    reshaped = prediction.reshape([B] + reshaper)
    ne = _normalised_entropy(reshaped)
    return torch.stack([ne, 1.0 - ne], dim=-1)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class NanoCTM(nn.Module):
    """
    Continuous Thought Machine — minimal, trainable implementation.

    The model's key property is iterative refinement: it runs the same input
    through T recurrent steps, updating its internal neuron states each time.
    This lets it allocate more computation to harder inputs.

    Key differences from a standard Transformer:
    - No token-to-token attention. The input is attended to via a single
      query derived from neuron synchronisation, not from each input token.
    - Per-neuron memory. Each neuron has its own history and its own linear
      model (NLM), enabling heterogeneous neuron behaviour to emerge.
    - Certainty-weighted output. Predictions at each step are weighted by
      how confident (low-entropy) the model is at that step.

    Args:
        config: a CTMConfig instance
    """

    # Class-level buffer type declarations — needed because register_buffer's
    # PyTorch type stub returns None, leaving Pyright unable to infer the type.
    idx_left_action:  torch.Tensor
    idx_right_action: torch.Tensor
    idx_left_out:     torch.Tensor
    idx_right_out:    torch.Tensor

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.config = config
        cfg = config

        # --- Backbone: integer token → dense embedding ---
        self.backbone = nn.Embedding(cfg.n_embedding, cfg.d_embedding)

        # --- KV projection: embedding → attention key/value space ---
        # Applied first so the positional encoding is added in the right space.
        self.kv_proj = nn.Sequential(
            nn.Linear(cfg.d_embedding, cfg.d_input),
            nn.LayerNorm(cfg.d_input),
        )

        # --- Positional encoding (added after kv_proj) ---
        self.pos_embedding = RotaryPositionalEmbedding(cfg.d_input)

        # --- Neuron-Level Model ---
        self.nlm = NLM(
            memory_length=cfg.memory_length,
            d_model=cfg.d_model,
            out_dims=2,        # 2 because NLM uses GLU: (B, N, 2) → (B, N, 1) → (B, N)
            dropout=cfg.dropout,
        )

        # --- Synapse network: (attn_out || activated_state) → new state ---
        self.synapses = SynapseNet(cfg.d_input, cfg.d_model, dropout=cfg.dropout)

        # --- Multi-head cross-attention ---
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.d_input,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        # --- Query projection: action-sync vector → attention query ---
        self.q_proj = nn.Linear(cfg.synch_rep_size_action, cfg.d_input)

        # --- Output projection: out-sync vector → per-step prediction logits ---
        self.output_proj = nn.Linear(cfg.synch_rep_size_out, cfg.out_dims)

        # --- Learnable decay parameters for synchronisation ---
        # Clamped to [0, 15] in forward pass → r = exp(-clamp(...)) ∈ (exp(-15), 1)
        self.decay_params_action = nn.Parameter(torch.zeros(cfg.synch_rep_size_action))
        self.decay_params_out    = nn.Parameter(torch.zeros(cfg.synch_rep_size_out))

        # --- Learnable initial recurrent state ---
        self.start_activated_state = nn.Parameter(
            torch.zeros(cfg.d_model).uniform_(-0.1, 0.1)
        )
        self.start_state_trace = nn.Parameter(
            torch.zeros(cfg.d_model, cfg.memory_length).uniform_(-0.1, 0.1)
        )

        # --- Neuron index buffers ---
        # Registered as buffers (not parameters): they move with .to(device) and
        # are saved in state_dict, but are not updated by the optimiser.
        device = torch.device("cpu")
        il_a, ir_a = _init_neuron_indices(
            cfg.neuron_select_type, "action", cfg.d_model, cfg.n_synch_action, device
        )
        il_o, ir_o = _init_neuron_indices(
            cfg.neuron_select_type, "out", cfg.d_model, cfg.n_synch_out, device
        )
        self.register_buffer("idx_left_action",  il_a)
        self.register_buffer("idx_right_action", ir_a)
        self.register_buffer("idx_left_out",     il_o)
        self.register_buffer("idx_right_out",    ir_o)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, seq_len) — integer indices {0, 1} (parity input after preprocessing)

        Returns:
            predictions: (B, out_dims, T) — logits at each thinking step
            certainties: (B, 2,       T) — [uncertainty, certainty] at each step
        """
        cfg = self.config
        B   = x.size(0)

        # ------------------------------------------------------------------
        # 1. Encode input into static key/value pairs
        #    These don't change across thinking iterations — the model attends
        #    to the same input representation at each of the T steps.
        # ------------------------------------------------------------------
        kv  = self.backbone(x).float()          # (B, seq_len, d_embedding)
        kv  = self.kv_proj(kv)                  # (B, seq_len, d_input)
        kv_t = kv.transpose(1, 2)               # (B, d_input, seq_len)
        pos  = self.pos_embedding(kv_t)         # (B, d_input, seq_len)
        kv   = (kv_t + pos).transpose(1, 2)     # (B, seq_len, d_input)

        # ------------------------------------------------------------------
        # 2. Initialise recurrent state
        #    .clone() is necessary because state_trace is modified each step.
        # ------------------------------------------------------------------
        state_trace     = self.start_state_trace.unsqueeze(0).expand(B, -1, -1).clone()
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1).clone()

        # ------------------------------------------------------------------
        # 3. Compute decay rates from learnable parameters
        #    Clamping keeps r in a stable range; without it the model can
        #    collapse to r≈0 (no memory) or r≈1 (no forgetting).
        # ------------------------------------------------------------------
        r_action = torch.exp(
            -self.decay_params_action.clamp(0.0, 15.0)
        ).unsqueeze(0).expand(B, -1)            # (B, synch_rep_size_action)

        r_out = torch.exp(
            -self.decay_params_out.clamp(0.0, 15.0)
        ).unsqueeze(0).expand(B, -1)            # (B, synch_rep_size_out)

        # ------------------------------------------------------------------
        # 4. Warm up out-sync accumulators (once, before the loop)
        #    The out-sync is initialised one step ahead of the loop so it
        #    already has a non-trivial value at iteration t=0.
        # ------------------------------------------------------------------
        _, da_out, db_out = compute_sync(
            activated_state, None, None, r_out,
            cfg.neuron_select_type, cfg.n_synch_out,
            self.idx_left_out, self.idx_right_out,
        )

        predictions = torch.empty(B, cfg.out_dims, cfg.iterations, device=x.device)
        certainties = torch.empty(B, 2,            cfg.iterations, device=x.device)
        da_action = db_action = None

        # ------------------------------------------------------------------
        # 5. Recurrent thinking loop
        # ------------------------------------------------------------------
        for t in range(cfg.iterations):

            # (a) Action-sync: pairwise neuron correlations → attention query
            sync_action, da_action, db_action = compute_sync(
                activated_state, da_action, db_action, r_action,
                cfg.neuron_select_type, cfg.n_synch_action,
                self.idx_left_action, self.idx_right_action,
            )
            q = self.q_proj(sync_action).unsqueeze(1)       # (B, 1, d_input)

            # (b) Cross-attend over the (fixed) input key/value pairs
            attn_out, _ = self.attention(q, kv, kv, need_weights=False)
            attn_out = attn_out.squeeze(1)                  # (B, d_input)

            # (c) Synapse network: merge attention context with current state
            pre_syn = torch.cat([attn_out, activated_state], dim=-1)
            state   = self.synapses(pre_syn)                # (B, d_model)

            # (d) Update rolling state trace: drop the oldest step, append new one
            #     NOTE: this allocates a new tensor every step — first optimisation
            #     target is replacing this with a circular buffer (write-index + roll).
            state_trace = torch.cat(
                [state_trace[:, :, 1:], state.unsqueeze(-1)], dim=-1
            )                                               # (B, d_model, memory_length)

            # (e) NLM: each neuron independently reads its own history
            activated_state = self.nlm(state_trace)         # (B, d_model)

            # (f) Out-sync: pairwise correlations → prediction logits
            sync_out, da_out, db_out = compute_sync(
                activated_state, da_out, db_out, r_out,
                cfg.neuron_select_type, cfg.n_synch_out,
                self.idx_left_out, self.idx_right_out,
            )
            pred      = self.output_proj(sync_out)                          # (B, out_dims)
            certainty = compute_certainty(pred, [cfg.sequence_length, 2])   # (B, 2)

            predictions[..., t] = pred
            certainties[..., t] = certainty

        return predictions, certainties


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def ctm_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    certainties: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    """
    Certainty-weighted cross-entropy loss, averaged over T thinking steps.

    The model is rewarded more for being confidently correct. At each step t,
    the CE loss for each sample is weighted by that sample's certainty score.
    This encourages the model to converge to a confident answer over iterations.

    Args:
        predictions: (B, seq_len * 2, T) — logits at each position and step
        targets:     (B, seq_len)         — ground-truth class indices {0, 1}
        certainties: (B, 2, T)            — [:, 0, :] = uncertainty, [:, 1, :] = certainty
        sequence_length: needed to reshape flat predictions

    Returns:
        scalar loss
    """
    B, _, T = predictions.shape
    total = torch.zeros(1, device=predictions.device)

    for t in range(T):
        # Reshape from (B, seq_len*2) → (B*seq_len, 2) for cross_entropy
        logits = predictions[:, :, t].reshape(B, sequence_length, 2)
        logits = logits.reshape(B * sequence_length, 2)
        tgts   = targets.reshape(B * sequence_length)

        ce = F.cross_entropy(logits, tgts, reduction='none')    # (B * seq_len,)
        ce = ce.reshape(B, sequence_length).mean(dim=-1)        # (B,)

        certainty_weight = certainties[:, 1, t]                 # (B,) — certainty at step t
        total = total + (certainty_weight * ce).mean()

    return total / T


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: Optional[CTMConfig] = None):
    """
    Baseline training loop for the parity task.

    Prints loss and certainty at t=0 vs t=T-1 every 100 steps.
    Certainty should grow from t=0 to t=T across training — the model learns
    to converge to a confident answer over its thinking iterations.
    """
    if config is None:
        config = CTMConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Config     : {config}\n")

    dataset    = ParityDataset(config.sequence_length, length=100_000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = NanoCTM(config).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}\n")

    model.train()
    for step, (vector, target) in enumerate(dataloader):
        vector = vector.to(device)
        target = target.to(device)

        # Convert ±1 input to {0, 1} indices for the embedding layer
        x = (vector == 1).long()

        predictions, certainties = model(x)
        loss = ctm_loss(predictions, target, certainties, config.sequence_length)

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 100 == 0:
            cert_t0 = certainties[:, 1, 0].mean().item()
            cert_tT = certainties[:, 1, -1].mean().item()
            print(
                f"step {step:5d} | loss {loss.item():.4f} "
                f"| certainty  t=0 {cert_t0:.3f} → t=T {cert_tT:.3f}"
            )

        if step >= 2000:
            break

    print("\nDone.")


if __name__ == "__main__":
    train()
