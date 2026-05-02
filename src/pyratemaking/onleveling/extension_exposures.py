"""Extension of exposures (W&M §5.3).

Re-rate every historical record using the *current* rating algorithm. The
result is the premium that would have been collected if every policy had been
written under today's rates. More accurate than :mod:`parallelogram` because
it captures rating-variable mix shifts and any non-uniform writing pattern,
but it requires the full rating algorithm and per-policy classification.

The rating algorithm is supplied as a callable. We do not assume any specific
factor structure (multiplicative GLM, additive offsets, capped, etc.) — that
is the user's domain knowledge.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def rate_under_algorithm(
    policies: pd.DataFrame,
    rating_algorithm: Callable[[pd.DataFrame], np.ndarray | pd.Series],
) -> pd.Series:
    """Apply ``rating_algorithm`` to ``policies`` and return premium per row.

    The algorithm receives the full DataFrame and must return an array-like of
    the same length. Internal validation only checks that the lengths match.
    """
    out = rating_algorithm(policies)
    out = np.asarray(out, dtype=float)
    if out.shape[0] != len(policies):
        raise ValueError(
            f"rating_algorithm returned {out.shape[0]} rows for {len(policies)} policies"
        )
    if (out < 0).any():
        raise ValueError("rating_algorithm produced negative premium")
    return pd.Series(out, index=policies.index, name="rerated_premium")


def extension_of_exposures(
    policies: pd.DataFrame,
    rating_algorithm: Callable[[pd.DataFrame], np.ndarray | pd.Series],
    *,
    ay_col: str = "policy_ay",
    collected_premium_col: str = "earned_premium",
) -> pd.DataFrame:
    """On-level historical premium by re-rating every policy.

    Parameters
    ----------
    policies : DataFrame
        Per-policy table with the rating variables that ``rating_algorithm``
        consumes, plus ``ay_col`` and ``collected_premium_col``.
    rating_algorithm : callable
        Maps a DataFrame to the premium that would be charged under today's
        rates. Vectorised: returns an array.
    ay_col, collected_premium_col : str

    Returns
    -------
    DataFrame indexed by AY with columns ``earned_premium``,
    ``on_level_premium``, ``on_level_factor`` (ratio of the two).
    """
    if ay_col not in policies.columns:
        raise KeyError(f"policies missing {ay_col!r}")
    if collected_premium_col not in policies.columns:
        raise KeyError(f"policies missing {collected_premium_col!r}")

    rerated = rate_under_algorithm(policies, rating_algorithm)
    work = policies[[ay_col, collected_premium_col]].copy()
    work["on_level_premium"] = rerated.to_numpy()

    grouped = work.groupby(ay_col).agg(
        earned_premium=(collected_premium_col, "sum"),
        on_level_premium=("on_level_premium", "sum"),
    )
    grouped["on_level_factor"] = (
        grouped["on_level_premium"] / grouped["earned_premium"]
    ).where(grouped["earned_premium"] > 0)
    grouped.index.name = "ay"
    return grouped
