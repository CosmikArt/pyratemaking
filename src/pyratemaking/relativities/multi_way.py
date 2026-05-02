"""Multi-way relativities derived from a fitted GLM.

Pulls relativity tables out of a :class:`~pyratemaking.glm.GLMResult` for
each requested categorical variable. Optionally rebases to a chosen level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyratemaking.glm.backend import GLMResult


def multi_way_relativities(
    fit: GLMResult,
    variables: list[str],
    *,
    rebase_to: dict[str, object] | None = None,
) -> dict[str, pd.Series]:
    """Return relativity tables for each requested variable.

    Parameters
    ----------
    fit : GLMResult
        A log-link GLM fit on at least the requested variables.
    variables : list[str]
        Categorical column names. Each must have been a categorical input.
    rebase_to : dict, optional
        Map of ``variable -> new base level``. The relativity at the new base
        is renormalised to 1.0.
    """
    rebase_to = rebase_to or {}
    out: dict[str, pd.Series] = {}
    for v in variables:
        rel = fit.relativities(v)
        if v in rebase_to:
            new_base = rebase_to[v]
            if new_base not in rel.index:
                raise ValueError(f"rebase target {new_base!r} not a level of {v!r}")
            rel = rel / float(rel.loc[new_base])
        out[v] = rel
    return out


def relativities_to_frame(
    relativities: dict[str, pd.Series],
) -> pd.DataFrame:
    """Stack a ``{variable: Series}`` dict into a single tidy frame."""
    rows = []
    for var, rel in relativities.items():
        for level, value in rel.items():
            rows.append({"variable": var, "level": level, "relativity": float(value)})
    return pd.DataFrame(rows)


def balance_principle_check(
    df: pd.DataFrame,
    relativities: dict[str, pd.Series],
    base_rate: float,
    *,
    exposure_col: str = "exposure",
    loss_col: str = "incurred_losses",
) -> pd.Series:
    """Compute the off-balance ratio after applying base × relativities (W&M §10).

    The balance principle requires the *total* rated premium under the new
    structure to match the actual loss + expense need. Returns a Series with
    ``rated_premium``, ``actual_losses``, and the ``off_balance`` ratio.
    """
    factors = np.ones(len(df))
    for var, rel in relativities.items():
        if var not in df.columns:
            raise KeyError(f"{var!r} not in df")
        rel_map = rel.astype(float).to_dict()
        if any(lvl not in rel_map for lvl in df[var].unique()):
            missing = [lvl for lvl in df[var].unique() if lvl not in rel_map]
            raise ValueError(f"unmapped levels for {var!r}: {missing}")
        factors *= df[var].map(rel_map).to_numpy(dtype=float)
    rated_premium = float((df[exposure_col] * base_rate * factors).sum())
    actual_losses = float(df[loss_col].sum())
    return pd.Series(
        {
            "rated_premium": rated_premium,
            "actual_losses": actual_losses,
            "off_balance": (actual_losses / rated_premium) if rated_premium > 0 else np.nan,
        }
    )
