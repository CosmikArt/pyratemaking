"""Increased limits factors (ILFs) and layer-loss costs (W&M §11.3).

Empirical ILFs come from a censored loss sample: ``ILF(L) = E[min(X, L)] /
E[min(X, B)]`` for limit ``L`` against basic limit ``B``. We construct ILFs
by Monte-Carlo from the raw loss data — no parametric severity model
required for v0.1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def layer_loss_cost(
    losses: np.ndarray | pd.Series,
    *,
    attachment: float,
    limit: float | None,
) -> float:
    """Average loss in the layer ``(attachment, attachment + limit]``.

    ``limit=None`` means an unlimited (per-occurrence excess) layer.
    """
    arr = np.asarray(losses, dtype=float)
    if attachment < 0:
        raise ValueError("attachment must be non-negative")
    payouts = np.maximum(arr - attachment, 0.0)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        payouts = np.minimum(payouts, limit)
    return float(payouts.mean())


def increased_limits_factor_table(
    losses: np.ndarray | pd.Series,
    *,
    basic_limit: float,
    limits: Sequence[float],
) -> pd.DataFrame:
    """Empirical ILF table at the requested limits.

    Returns a DataFrame indexed by ``limit`` with columns ``avg_capped_loss``
    and ``ilf`` (``= avg_capped_loss / avg_capped_loss[basic_limit]``).
    """
    arr = np.asarray(losses, dtype=float)
    if (arr < 0).any():
        raise ValueError("losses contain negative values")
    base_avg = float(np.minimum(arr, basic_limit).mean())
    if base_avg <= 0:
        raise ValueError("basic-limit average loss is zero")

    rows = []
    for L in sorted(set([basic_limit, *limits])):
        avg = float(np.minimum(arr, L).mean())
        rows.append({"limit": L, "avg_capped_loss": avg, "ilf": avg / base_avg})
    return pd.DataFrame(rows).set_index("limit")
