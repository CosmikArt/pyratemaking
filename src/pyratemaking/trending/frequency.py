"""Frequency trending.

Frequency per AY = claim count / earned exposure. Multiplicative form is the
default (constant percentage change). Weighted by exposure when fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyratemaking.trending.core import Trend, fit_trend


def frequency_trend(
    claim_count: pd.Series,
    earned_exposure: pd.Series,
    *,
    times: pd.Series | None = None,
    kind: str = "multiplicative",
    weight_by_exposure: bool = True,
) -> Trend:
    """Fit a trend to claim frequency by period."""
    if not claim_count.index.equals(earned_exposure.index):
        claim_count, earned_exposure = claim_count.align(earned_exposure, join="inner")
    counts = claim_count.to_numpy(dtype=float)
    exp_arr = earned_exposure.to_numpy(dtype=float)
    freq = np.where(exp_arr > 0, counts / np.where(exp_arr > 0, exp_arr, 1), np.nan)
    freq_series = pd.Series(freq, index=claim_count.index, name="frequency").dropna()
    if times is None:
        times = pd.Series(
            freq_series.index.to_numpy(dtype=float), index=freq_series.index
        )
    weights = exp_arr[~np.isnan(freq)] if weight_by_exposure else None
    return fit_trend(freq_series, times.loc[freq_series.index], kind=kind, weights=weights)
