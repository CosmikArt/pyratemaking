"""Basic-limits losses: cap individual claims at a stated limit (W&M §11.2).

The capped loss series feeds into rate-level indication and classification
so the result is not driven by a few large losses. The amount above the
limit is priced separately via increased-limits factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cap_at_limit(losses: pd.Series | np.ndarray, basic_limit: float) -> np.ndarray:
    """Element-wise cap at ``basic_limit``."""
    if basic_limit <= 0:
        raise ValueError("basic_limit must be positive")
    arr = np.asarray(losses, dtype=float)
    return np.minimum(arr, basic_limit)


def basic_limits_losses(
    claims: pd.DataFrame,
    *,
    loss_col: str = "claim_amount",
    basic_limit: float,
    ay_col: str | None = "policy_ay",
) -> pd.DataFrame:
    """Cap each claim and return the capped frame plus excess column.

    Parameters
    ----------
    claims : DataFrame
    loss_col : str
        Column with raw loss amounts.
    basic_limit : float
        Cap level. Per-claim losses above this go into ``excess_amount``.
    ay_col : str, optional
        When provided, returns a per-AY summary as ``.attrs["summary"]``.
    """
    if loss_col not in claims.columns:
        raise KeyError(f"{loss_col!r} not in claims")

    out = claims.copy()
    raw = out[loss_col].to_numpy(dtype=float)
    capped = cap_at_limit(raw, basic_limit)
    out["basic_limits_loss"] = capped
    out["excess_amount"] = np.maximum(raw - basic_limit, 0.0)

    if ay_col is not None and ay_col in out.columns:
        summary = out.groupby(ay_col).agg(
            total_loss=(loss_col, "sum"),
            basic_limits_loss=("basic_limits_loss", "sum"),
            excess_loss=("excess_amount", "sum"),
            claim_count=(loss_col, "count"),
            n_capped=("excess_amount", lambda s: int((s > 0).sum())),
        )
        out.attrs["summary"] = summary
    return out
