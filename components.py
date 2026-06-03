import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuronModel(nn.Module):
    """Per-neuron linear model: maps each neuron's rolling history → next activation."""

    def __init__(self, memory_length: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(memory_length)

        bound       = 1.0 / math.sqrt(memory_length + 2)
        self.weight = nn.Parameter(torch.empty(memory_length, 2, d_model).uniform_(-bound, bound))
        self.bias   = nn.Parameter(torch.zeros(1, d_model, 2))
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, input_trace: torch.Tensor) -> torch.Tensor:
        # input_trace: (B, d_model, memory_length)
        x = self.dropout(input_trace)
        x = self.layer_norm(x)
        # einsum: each of the d_model neurons applies its own (M→2) linear to its own history
        x = torch.einsum("BNM,MON->BNO", x, self.weight) + self.bias  # (B, d_model, 2)
        return F.glu(x, dim=-1).squeeze(-1) / self.temperature          # (B, d_model)


class SynapseNet(nn.Module):
    """Maps cat(attn_out, activated_state) → new pre-NLM internal state."""

    def __init__(self, d_input: int, d_model: int, dropout: float = 0.0, net_type: str = "dense"):
        super().__init__()
        if net_type == "dense":
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_input + d_model, d_model * 2),
                nn.GLU(dim=-1),
                nn.LayerNorm(d_model),
            )
        elif net_type == "unet":
            raise NotImplementedError("UNet synapse is not implemented yet.")
        else:
            raise ValueError(f"Unknown net_type: {net_type!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
