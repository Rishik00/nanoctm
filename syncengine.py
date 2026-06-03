import torch
from typing import Optional, Tuple


def compute_synchronization(
    activated_state: torch.Tensor,
    ema_numer: Optional[torch.Tensor],
    ema_denom: Optional[torch.Tensor],
    decay_rate: torch.Tensor,
    n_synch: int,
    idx_left: torch.Tensor,
    idx_right: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    select_left  = activated_state[:, idx_left]
    select_right = activated_state[:, idx_right]
    outer_prod   = select_left.unsqueeze(2) * select_right.unsqueeze(1)  # (B, n, n)
    row_idx, col_idx = torch.triu_indices(n_synch, n_synch, device=activated_state.device)
    pairwise = outer_prod[:, row_idx, col_idx]                            # (B, rep_size)

    if ema_numer is None or ema_denom is None:
        ema_numer = pairwise
        ema_denom = torch.ones_like(pairwise)
    else:
        ema_numer = decay_rate * ema_numer + pairwise
        ema_denom = decay_rate * ema_denom + 1.0

    return ema_numer / ema_denom.sqrt(), ema_numer, ema_denom

__all__ = ["compute_synchronization"]
