"""One-way relativity tables: pure premium per level relative to base.

A starting point for diagnostics — captures unbalanced exposure and
correlated rating variables, but does not adjust for them. For final
relativities, use :func:`multi_way_relativities` (GLM-based).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def one_way_relativities(
    df: pd.DataFrame,
    variable: str,
    *,
    exposure_col: str = "exposure",
    loss_col: str = "incurred_losses",
    count_col: str | None = None,
    base_level: str | int | float | None = None,
) -> pd.DataFrame:
    """Pure-premium relativities for a single variable.

    Returns a frame indexed by level with columns ``exposure``,
    ``losses``, ``pure_premium``, ``relativity``, plus ``frequency`` and
    ``severity`` when ``count_col`` is provided.
    """
    if variable not in df.columns:
        raise KeyError(f"variable {variable!r} not in dataframe")

    agg = {
        "exposure": (exposure_col, "sum"),
        "losses": (loss_col, "sum"),
    }
    if count_col is not None:
        agg["count"] = (count_col, "sum")

    grouped = df.groupby(variable, dropna=False).agg(**agg)
    grouped["pure_premium"] = grouped["losses"] / grouped["exposure"].where(grouped["exposure"] > 0)

    if count_col is not None:
        grouped["frequency"] = grouped["count"] / grouped["exposure"].where(grouped["exposure"] > 0)
        grouped["severity"] = grouped["losses"] / grouped["count"].where(grouped["count"] > 0)

    if base_level is None:
        base_level = grouped["exposure"].idxmax()
    if base_level not in grouped.index:
        raise ValueError(f"base level {base_level!r} not in {variable!r}")

    base_pp = float(grouped.loc[base_level, "pure_premium"])
    if base_pp <= 0 or np.isnan(base_pp):
        raise ValueError(f"base level {base_level!r} has zero/missing pure premium")
    grouped["relativity"] = grouped["pure_premium"] / base_pp
    grouped.index.name = variable
    return grouped
