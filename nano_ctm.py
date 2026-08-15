"""
nano_ctm.py
-----------
A clean, minimal implementation of the Continuous Thought Machine (CTM),
in the spirit of nanoGPT: one file, readable, trainable from scratch.

Architecture in one paragraph:
    A frozen pre-trained bidirectional encoder (DistilBERT by default) encodes
    the input into contextual token representations used as static key/value pairs.
    The CTM then runs T recurrent "thinking" iterations. Each iteration uses
    pairwise neuron correlations (synchronisation) to form an attention query,
    attends over the frozen KVs, feeds the result through a synapse network to
    produce a new internal state, and updates each neuron's personal history.
    A per-neuron linear model (NLM) reads that history to produce the next
    activated state. A separate out-synchronisation projects the final neuron
    correlations to per-step predictions. Certainty (1 - normalised entropy)
    weights the loss across iterations.

Task: SST-2 binary sentiment classification (GLUE benchmark).

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
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CTMConfig:
    """All hyperparameters in one place. Change here, nowhere else."""

    # Frozen backbone (bidirectional encoder)
    backbone_name: str = "distilbert-base-uncased"
    d_backbone: int = 768       # Must match the backbone's hidden size
    max_seq_len: int = 128      # Truncation / padding length for the tokenizer

    # Task
    n_classes: int = 2

    # Model dimensions
    d_model: int = 256          # Number of neurons (size of the internal state vector)
    d_input: int = 256          # Dimension used for attention Q/K/V

    # Attention
    num_heads: int = 8

    # Memory / NLM
    memory_length: int = 20     # History window length per neuron

    # Synchronisation
    n_synch_out: int = 64       # Neurons used for output-sync (drives predictions)
    n_synch_action: int = 64    # Neurons used for action-sync (drives attention query)
    neuron_select_type: Literal["first-last", "random", "random-pairing"] = "first-last"

    # Training
    dropout: float = 0.1
    iterations: int = 20        # T: number of thinking steps per forward pass

    # Derived — set automatically, do not pass manually
    out_dims: int = field(init=False)
    synch_rep_size_action: int = field(init=False)
    synch_rep_size_out: int = field(init=False)

    def __post_init__(self):
        self.out_dims = self.n_classes
        self.synch_rep_size_action = _sync_rep_size(
            self.neuron_select_type, self.n_synch_action
        )
        self.synch_rep_size_out = _sync_rep_size(
            self.neuron_select_type, self.n_synch_out
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SST2Dataset(Dataset):
    """
    SST-2 binary sentiment classification (GLUE benchmark).

    Tokenizes all sentences upfront and stores them as tensors so the DataLoader
    workers do zero NLP work. Labels: 0 = negative, 1 = positive.

    Args:
        split:         "train" or "validation"
        backbone_name: tokenizer to use — should match the CTM backbone
        max_length:    truncation / padding target
    """

    def __init__(
        self,
        split: str = "train",
        backbone_name: str = "distilbert-base-uncased",
        max_length: int = 128,
    ):
        from datasets import load_dataset

        raw = load_dataset("nyu-mll/glue", "sst2", split=split)
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        enc = tokenizer(
            list(raw["sentence"]),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]           # (N, max_length)
        self.attention_mask = enc["attention_mask"]  # (N, max_length)
        self.labels = torch.tensor(raw["label"], dtype=torch.long)  # (N,)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


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
        self.dropout = nn.Dropout(dropout)
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
        x = self.layernorm(x)                          # (B, N, M)
        x = torch.einsum("BNM,MON->BNO", x, self.W) + self.b  # (B, N, 2)
        x = F.glu(x, dim=-1)                           # (B, N, 1)
        return x.squeeze(-1) / self.temperature        # (B, N)


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
            nn.Linear(d_input + d_model, d_model * 2),  # × 2 because GLU halves it
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
        return idx, idx  # left == right for first-last

    elif neuron_select_type in ("random", "random-pairing"):
        left = torch.from_numpy(
            np.random.choice(d_model, size=n_synch, replace=False)
        ).to(device)
        right = torch.from_numpy(
            np.random.choice(d_model, size=n_synch, replace=False)
        ).to(device)
        return left, right

    raise ValueError(f"Unknown neuron_select_type: {neuron_select_type!r}")


def compute_sync(
    activated_state: torch.Tensor,
    decay_alpha: Optional[torch.Tensor],
    decay_beta: Optional[torch.Tensor],
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
        pairwise = activated_state[:, idx_left] * activated_state[:, idx_right]
    else:
        sel_left = activated_state[:, idx_left]
        sel_right = activated_state[:, idx_right]
        outer = sel_left.unsqueeze(2) * sel_right.unsqueeze(1)  # (B, n_synch, n_synch)
        i, j = torch.triu_indices(n_synch, n_synch, device=activated_state.device)
        pairwise = outer[:, i, j]  # (B, synch_rep_size)

    if decay_alpha is None or decay_beta is None:
        decay_alpha = pairwise
        decay_beta = torch.ones_like(pairwise)
    else:
        decay_alpha = r * decay_alpha + pairwise
        decay_beta = r * decay_beta + 1.0

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
    probs = logits.softmax(dim=-1)
    log_p = logits.log_softmax(dim=-1)
    entropy = -(probs * log_p).sum(dim=-1)
    max_ent = math.log(logits.size(-1))
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
                    e.g. [n_classes] for classification
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
    Continuous Thought Machine with a frozen bidirectional encoder backbone.

    The backbone (e.g. DistilBERT) encodes the input once into contextual
    token representations. The CTM then runs T recurrent thinking iterations,
    attending over those fixed representations while updating its internal
    neuron states. This separates perception (backbone) from reasoning (CTM).

    Args:
        config: a CTMConfig instance
    """

    # Class-level buffer type declarations — needed because register_buffer's
    # PyTorch type stub returns None, leaving Pyright unable to infer the type.
    idx_left_action: torch.Tensor
    idx_right_action: torch.Tensor
    idx_left_out: torch.Tensor
    idx_right_out: torch.Tensor

    def __init__(self, config: CTMConfig):
        super().__init__()
        self.config = config
        cfg = config

        # --- Frozen backbone: token ids → contextual hidden states ---
        backbone = AutoModel.from_pretrained(cfg.backbone_name)
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone

        # --- KV projection: d_backbone → d_input ---
        # Applied to the backbone's last hidden states before cross-attention.
        self.kv_proj = nn.Sequential(
            nn.Linear(cfg.d_backbone, cfg.d_input),
            nn.LayerNorm(cfg.d_input),
        )

        # --- Neuron-Level Model ---
        self.nlm = NLM(
            memory_length=cfg.memory_length,
            d_model=cfg.d_model,
            out_dims=2,  # GLU: (B, N, 2) → (B, N, 1) → (B, N)
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
        self.decay_params_action = nn.Parameter(torch.zeros(cfg.synch_rep_size_action))
        self.decay_params_out = nn.Parameter(torch.zeros(cfg.synch_rep_size_out))

        # --- Learnable initial recurrent state ---
        self.start_activated_state = nn.Parameter(
            torch.zeros(cfg.d_model).uniform_(-0.1, 0.1)
        )
        self.start_state_trace = nn.Parameter(
            torch.zeros(cfg.d_model, cfg.memory_length).uniform_(-0.1, 0.1)
        )

        # --- Neuron index buffers ---
        device = torch.device("cpu")
        il_a, ir_a = _init_neuron_indices(
            cfg.neuron_select_type, "action", cfg.d_model, cfg.n_synch_action, device
        )
        il_o, ir_o = _init_neuron_indices(
            cfg.neuron_select_type, "out", cfg.d_model, cfg.n_synch_out, device
        )
        self.register_buffer("idx_left_action", il_a)
        self.register_buffer("idx_right_action", ir_a)
        self.register_buffer("idx_left_out", il_o)
        self.register_buffer("idx_right_out", ir_o)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      (B, seq_len) — token ids from the tokenizer
            attention_mask: (B, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            predictions: (B, n_classes, T) — logits at each thinking step
            certainties: (B, 2,        T) — [uncertainty, certainty] at each step
        """
        cfg = self.config
        B = input_ids.size(0)

        # ------------------------------------------------------------------
        # 1. Encode input with the frozen backbone
        #    last_hidden_state: (B, seq_len, d_backbone)
        #    The backbone already encodes positional information internally,
        #    so no additional positional encoding is needed here.
        # ------------------------------------------------------------------
        with torch.no_grad():
            backbone_out = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask
            )
        kv = self.kv_proj(backbone_out.last_hidden_state)  # (B, seq_len, d_input)

        # PyTorch MHA key_padding_mask: True = ignore that position
        key_padding_mask = attention_mask == 0  # (B, seq_len)

        # ------------------------------------------------------------------
        # 2. Initialise recurrent state
        # ------------------------------------------------------------------
        state_trace = self.start_state_trace.unsqueeze(0).expand(B, -1, -1).clone()
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1).clone()

        # ------------------------------------------------------------------
        # 3. Compute decay rates from learnable parameters
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 4. Warm up out-sync accumulators (once, before the loop)
        # ------------------------------------------------------------------
        _, da_out, db_out = compute_sync(
            activated_state,
            None,
            None,
            r_out,
            cfg.neuron_select_type,
            cfg.n_synch_out,
            self.idx_left_out,
            self.idx_right_out,
        )

        predictions = torch.empty(B, cfg.out_dims, cfg.iterations, device=input_ids.device)
        certainties = torch.empty(B, 2, cfg.iterations, device=input_ids.device)
        da_action = db_action = None

        # ------------------------------------------------------------------
        # 5. Recurrent thinking loop
        # ------------------------------------------------------------------
        for t in range(cfg.iterations):
            # (a) Action-sync: pairwise neuron correlations → attention query
            sync_action, da_action, db_action = compute_sync(
                activated_state,
                da_action,
                db_action,
                r_action,
                cfg.neuron_select_type,
                cfg.n_synch_action,
                self.idx_left_action,
                self.idx_right_action,
            )
            q = self.q_proj(sync_action).unsqueeze(1)  # (B, 1, d_input)

            # (b) Cross-attend over the frozen backbone KV pairs
            #     key_padding_mask masks out padding tokens
            attn_out, _ = self.attention(
                q, kv, kv,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            attn_out = attn_out.squeeze(1)  # (B, d_input)

            # (c) Synapse network: merge attention context with current state
            pre_syn = torch.cat([attn_out, activated_state], dim=-1)
            state = self.synapses(pre_syn)  # (B, d_model)

            # (d) Update rolling state trace: drop oldest step, append new one
            state_trace = torch.cat(
                [state_trace[:, :, 1:], state.unsqueeze(-1)], dim=-1
            )  # (B, d_model, memory_length)

            # (e) NLM: each neuron independently reads its own history
            activated_state = self.nlm(state_trace)  # (B, d_model)

            # (f) Out-sync: pairwise correlations → prediction logits
            sync_out, da_out, db_out = compute_sync(
                activated_state,
                da_out,
                db_out,
                r_out,
                cfg.neuron_select_type,
                cfg.n_synch_out,
                self.idx_left_out,
                self.idx_right_out,
            )
            pred = self.output_proj(sync_out)                    # (B, n_classes)
            certainty = compute_certainty(pred, [cfg.n_classes]) # (B, 2)

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
) -> torch.Tensor:
    """
    Certainty-weighted cross-entropy loss, averaged over T thinking steps.

    The model is rewarded more for being confidently correct. At each step t,
    the CE loss is weighted by that step's certainty score. Softmax-normalising
    the certainty weights prevents the model from escaping the loss by being
    uniformly uncertain.

    Args:
        predictions: (B, n_classes, T) — logits at each thinking step
        targets:     (B,)              — ground-truth class indices
        certainties: (B, 2, T)         — [:, 0, :] = uncertainty, [:, 1, :] = certainty

    Returns:
        scalar loss
    """
    B, _, T = predictions.shape

    cert_weights = F.softmax(certainties[:, 1, :], dim=-1)  # (B, T)

    total = torch.zeros(1, device=predictions.device)
    for t in range(T):
        logits = predictions[:, :, t]                              # (B, n_classes)
        ce = F.cross_entropy(logits, targets, reduction="none")    # (B,)
        total = total + (cert_weights[:, t] * ce).mean()

    return total


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(config: Optional[CTMConfig] = None, fast: bool = True):
    """
    Training loop for SST-2 sentiment classification.

    Prints loss, accuracy, and certainty at t=0 vs t=T-1 every 100 steps.
    Certainty should grow from t=0 to t=T across training — the model learns
    to converge to a confident answer over its thinking iterations.

    Args:
        fast: use bfloat16 autocast + fused AdamW when CUDA is available.
    """
    if config is None:
        config = CTMConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = device.type == "cuda"

    if fast and on_cuda:
        torch.set_float32_matmul_precision("high")

    batch_size = 64 if on_cuda else 16

    print(f"Device     : {device}  (fast={fast and on_cuda})")
    print(f"Batch      : {batch_size}")
    print(f"Config     : {config}\n")

    print("Loading SST-2 dataset...")
    train_dataset = SST2Dataset("train", config.backbone_name, config.max_seq_len)
    val_dataset = SST2Dataset("validation", config.backbone_name, config.max_seq_len)
    print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=on_cuda,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=on_cuda,
        persistent_workers=True,
    )

    model = NanoCTM(config).to(device)

    # Only optimise CTM parameters — backbone is frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable,
        lr=1e-3,
        weight_decay=0.01,
        fused=(fast and on_cuda),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    print(f"Parameters : {total_params:,} total  /  {trainable_params:,} trainable\n")

    model.train()
    for step, (input_ids, attention_mask, target) in enumerate(train_loader):
        input_ids = input_ids.to(device, non_blocking=on_cuda)
        attention_mask = attention_mask.to(device, non_blocking=on_cuda)
        target = target.to(device, non_blocking=on_cuda)

        if fast and on_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions, certainties = model(input_ids, attention_mask)
                loss = ctm_loss(predictions, target, certainties)
        else:
            predictions, certainties = model(input_ids, attention_mask)
            loss = ctm_loss(predictions, target, certainties)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()

        if step % 100 == 0:
            # Accuracy: use the final thinking step's prediction
            acc = (predictions[:, :, -1].argmax(dim=-1) == target).float().mean().item()
            cert_t0 = certainties[:, 1, 0].mean().item()
            cert_tT = certainties[:, 1, -1].mean().item()
            print(
                f"step {step:5d} | loss {loss.item():.4f} | acc {acc:.3f} "
                f"| certainty  t=0 {cert_t0:.3f} → t=T {cert_tT:.3f}"
            )

        if step >= 2000:
            break

    # --- Quick validation pass ---
    print("\nRunning validation...")
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for input_ids, attention_mask, target in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)
            predictions, _ = model(input_ids, attention_mask)
            preds = predictions[:, :, -1].argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    print(f"Val accuracy: {correct / total:.4f}  ({correct}/{total})")

    print("\nDone.")


if __name__ == "__main__":
    train()
