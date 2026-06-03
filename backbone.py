# Supported backbone_type values:
#   "parity_embedding"   — learnable embedding, for the parity task
#   "bert"               — BERT-base, frozen by default
#   "gpt2"               — GPT-2 small, frozen by default

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# HuggingFace wrappers
# ---------------------------------------------------------------------------

class BERTBackbone(nn.Module):
    """
    BERT-base as a frozen (or trainable) sequence encoder.

    Input:  (B, seq_len) — int token IDs
    Output: (B, seq_len, 768)
    """
    def __init__(self, freeze: bool = True):
        super().__init__()
        from transformers import AutoModel      # lazy: only loads if BERT is used
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).last_hidden_state  # (B, seq_len, 768)


class GPT2Backbone(nn.Module):
    """
    GPT-2 small as a frozen (or trainable) sequence encoder.

    Input:  (B, seq_len) — int token IDs
    Output: (B, seq_len, 768)
    """
    def __init__(self, freeze: bool = True):
        super().__init__()
        # Review: aren't we supposed to use automodelforcausallm? 
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("gpt2")


        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).last_hidden_state  # (B, seq_len, 768)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def return_backbone(config) -> nn.Module:
    btype = config.backbone_type
    freeze = getattr(config, "freeze_backbone", True)

    if btype == "parity_embedding":
        return nn.Embedding(config.n_embedding, config.d_embedding)

    if btype == "bert":
        return BERTBackbone(freeze=freeze)

    if btype == "gpt2":
        return GPT2Backbone(freeze=freeze)

    if btype.startswith("resnet18-"):
        return NotImplemented("Resnet is not implemented yet!")

    raise ValueError(f"Unknown backbone_type: {btype!r}")
