"""Severity trending.

Average severity per AY = total incurred losses / claim count.
Weighted by claim count by default — heavier years carry more information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyratemaking.trending.core import Trend, fit_trend


def severity_trend(
    incurred_losses: pd.Series,
    claim_count: pd.Series,
    *,
    times: pd.Series | None = None,
    kind: str = "multiplicative",
    weight_by_count: bool = True,
) -> Trend:
    """Fit a trend to average severity by period.

    Parameters
    ----------
    incurred_losses, claim_count : Series
        Aligned by index (period).
    times : Series, optional
        Time index. Defaults to the shared index.
    kind : str
        Trend form (see :func:`pyratemaking.trending.fit_trend`).
    weight_by_count : bool, default True
        Weight the OLS fit by claim count.
    """
    if not incurred_losses.index.equals(claim_count.index):
        incurred_losses, claim_count = incurred_losses.align(claim_count, join="inner")
    counts = claim_count.to_numpy(dtype=float)
    losses = incurred_losses.to_numpy(dtype=float)
    severity = np.where(counts > 0, losses / np.where(counts > 0, counts, 1), np.nan)
    sev_series = pd.Series(severity, index=incurred_losses.index, name="avg_severity")
    sev_series = sev_series.dropna()
    if times is None:
        times = pd.Series(sev_series.index.to_numpy(dtype=float), index=sev_series.index)
    weights = counts[~np.isnan(severity)] if weight_by_count else None
    return fit_trend(sev_series, times.loc[sev_series.index], kind=kind, weights=weights)
