import math
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
from backbone import return_backbone
from components import NeuronModel, SynapseNet
from syncengine import compute_synchronization


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CTMConfig:
    # Backbone
    backbone_type: str = "parity_embedding"
    n_embedding:   int = 2    # parity: vocab size {0, 1}
    d_embedding:   int = 256  # parity: embedding output dim

    # Sequence
    max_seq_len: int = 32

    # Core dimensions
    d_model:   int = 256
    d_input:   int = 256  # attention embed dim — usually == d_model
    num_heads: int = 8

    # Memory
    memory_length: int = 20

    # Synchronization
    n_synch_action: int = 64
    n_synch_out:    int = 64

    # Training
    iterations: int   = 20
    dropout:    float = 0.1

    # Hardware
    use_flash:       bool = False
    use_triton:      bool = False  # reserved — Triton NLM kernel not yet implemented
    freeze_backbone: bool = True   # freeze backbone weights, only train CTM parts

    # Derived — set in __post_init__, do not pass manually
    d_backbone:            int = field(init=False)
    out_dims:              int = field(init=False)
    synch_rep_size_action: int = field(init=False)
    synch_rep_size_out:    int = field(init=False)

    def __post_init__(self):
        resnet18_layer_dims = {1: 64, 2: 128, 3: 256, 4: 512}
        if self.backbone_type == "parity_embedding":
            self.d_backbone = self.d_embedding
        elif self.backbone_type in ("bert", "gpt2"):
            self.d_backbone = 768
        elif self.backbone_type.startswith("resnet18-"):
            layer = int(self.backbone_type.split("-")[1])
            if layer not in resnet18_layer_dims:
                raise ValueError(f"resnet18 layer must be 1–4, got {layer}")
            self.d_backbone = resnet18_layer_dims[layer]
        else:
            raise ValueError(f"Unknown backbone_type: {self.backbone_type!r}")
        self.out_dims   = self.max_seq_len * 2
        self.synch_rep_size_action = (self.n_synch_action * (self.n_synch_action + 1)) // 2
        self.synch_rep_size_out    = (self.n_synch_out    * (self.n_synch_out    + 1)) // 2


# ---------------------------------------------------------------------------
# Certainty helpers (model-level — normalised entropy of per-step predictions)
# ---------------------------------------------------------------------------

def normalised_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs   = logits.softmax(dim=-1)
    entropy = -(probs * logits.log_softmax(dim=-1)).sum(dim=-1)
    norm_ent = entropy / math.log(logits.size(-1))
    if logits.dim() > 2:
        norm_ent = norm_ent.flatten(1).mean(-1)
    return norm_ent


def compute_certainty(pred: torch.Tensor, seq_len: int) -> torch.Tensor:
    B = pred.size(0)
    ne = normalised_entropy(pred.reshape(B, seq_len, 2))
    return torch.stack([ne, 1.0 - ne], dim=-1)  # (B, 2): [uncertainty, certainty]


# ---------------------------------------------------------------------------
# CTM
# ---------------------------------------------------------------------------

class CTM(nn.Module):
    """
    Continuous Thought Machine.

    Takes raw input, runs T recurrent thinking steps, and returns per-step
    predictions and certainty scores. The backbone is fixed across all steps;
    only the internal neuron state evolves.
    """

    idx_left_action:  torch.Tensor
    idx_right_action: torch.Tensor
    idx_left_out:     torch.Tensor
    idx_right_out:    torch.Tensor

    def __init__(self, cfg: CTMConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = return_backbone(cfg)

        self.kv_proj = nn.Sequential(
            nn.Linear(cfg.d_backbone, cfg.d_input),
            nn.LayerNorm(cfg.d_input),
        )

        self.nlm      = NeuronModel(cfg.memory_length, cfg.d_model, dropout=cfg.dropout)
        self.synapses = SynapseNet(cfg.d_input, cfg.d_model, dropout=cfg.dropout)

        self.attention = nn.MultiheadAttention(
            cfg.d_input, cfg.num_heads, dropout=cfg.dropout, batch_first=True,
        )

        self.q_proj      = nn.Linear(cfg.synch_rep_size_action, cfg.d_input)
        self.output_proj = nn.Linear(cfg.synch_rep_size_out,    cfg.out_dims)

        self.decay_action = nn.Parameter(torch.zeros(cfg.synch_rep_size_action))
        self.decay_out    = nn.Parameter(torch.zeros(cfg.synch_rep_size_out))

        self.start_act   = nn.Parameter(torch.zeros(cfg.d_model).uniform_(-0.1, 0.1))
        self.start_trace = nn.Parameter(torch.zeros(cfg.d_model, cfg.memory_length).uniform_(-0.1, 0.1))

        n_action = cfg.n_synch_action
        n_out    = cfg.n_synch_out
        self.register_buffer("idx_left_action",  torch.arange(cfg.d_model - n_action, cfg.d_model))
        self.register_buffer("idx_right_action", torch.arange(cfg.d_model - n_action, cfg.d_model))
        self.register_buffer("idx_left_out",     torch.arange(0, n_out))
        self.register_buffer("idx_right_out",    torch.arange(0, n_out))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, seq_len) integer token indices

        Returns:
            preds:  (B, out_dims, T)
            certs:  (B, 2, T)  — [:, 0, :] uncertainty, [:, 1, :] certainty
        """
        cfg = self.cfg
        features = self.backbone(x)                          # (B, seq_len, d_backbone)
        B  = features.size(0)
        kv = self.kv_proj(features)                          # (B, seq_len, d_input)

        state_trace     = self.start_trace.unsqueeze(0).expand(B, -1, -1).clone()
        activated_state = self.start_act.unsqueeze(0).expand(B, -1).clone()

        decay_rate_action = torch.exp(-self.decay_action.clamp(0, 15)).unsqueeze(0).expand(B, -1)
        decay_rate_out    = torch.exp(-self.decay_out.clamp(0, 15)).unsqueeze(0).expand(B, -1)

        # Warm up out-sync with the initial state so t=0 already has context
        _, ema_numer_out, ema_denom_out = compute_synchronization(
            activated_state, None, None, decay_rate_out,
            cfg.n_synch_out, self.idx_left_out, self.idx_right_out,
        )

        preds = torch.empty(B, cfg.out_dims, cfg.iterations, device=x.device)
        certs = torch.empty(B, 2,            cfg.iterations, device=x.device)
        ema_numer_act = ema_denom_act = None

        for t in range(cfg.iterations):
            # (a) Action-sync → attention query
            sync_action, ema_numer_act, ema_denom_act = compute_synchronization(
                activated_state, ema_numer_act, ema_denom_act, decay_rate_action,
                cfg.n_synch_action, self.idx_left_action, self.idx_right_action,
            )
            q = self.q_proj(sync_action).unsqueeze(1)        # (B, 1, d_input)

            # (b) Cross-attend over the fixed input KVs
            attn_out, _ = self.attention(q, kv, kv, need_weights=False)
            attn_out = attn_out.squeeze(1)                   # (B, d_input)

            # (c) Synapse: merge attention context with current neuron state
            state = self.synapses(torch.cat([attn_out, activated_state], dim=-1))

            # (d) Roll state trace: drop oldest step, append new state
            state_trace = torch.cat([state_trace[:, :, 1:], state.unsqueeze(-1)], dim=-1)

            # (e) NLM: each neuron reads its own history → new activated state
            activated_state = self.nlm(state_trace)          # (B, d_model)

            # (f) Out-sync → prediction logits + certainty
            sync_out, ema_numer_out, ema_denom_out = compute_synchronization(
                activated_state, ema_numer_out, ema_denom_out, decay_rate_out,
                cfg.n_synch_out, self.idx_left_out, self.idx_right_out,
            )
            pred = self.output_proj(sync_out)                # (B, out_dims)
            preds[..., t] = pred
            certs[..., t] = compute_certainty(pred, cfg.max_seq_len)

        return preds, certs


__all__ = ["CTM", "CTMConfig"]
