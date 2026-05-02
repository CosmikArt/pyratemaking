"""Pure-premium trending.

Pure premium per AY = incurred losses / earned exposure. Equivalent to
fitting severity and frequency separately and multiplying their projected
values, when the multiplicative form is used and there is no exposure
mix shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyratemaking.trending.core import Trend, fit_trend


def pure_premium_trend(
    incurred_losses: pd.Series,
    earned_exposure: pd.Series,
    *,
    times: pd.Series | None = None,
    kind: str = "multiplicative",
    weight_by_exposure: bool = True,
) -> Trend:
    """Fit a trend to pure premium by period."""
    if not incurred_losses.index.equals(earned_exposure.index):
        incurred_losses, earned_exposure = incurred_losses.align(earned_exposure, join="inner")
    losses = incurred_losses.to_numpy(dtype=float)
    exp_arr = earned_exposure.to_numpy(dtype=float)
    pp = np.where(exp_arr > 0, losses / np.where(exp_arr > 0, exp_arr, 1), np.nan)
    pp_series = pd.Series(pp, index=incurred_losses.index, name="pure_premium").dropna()
    if times is None:
        times = pd.Series(pp_series.index.to_numpy(dtype=float), index=pp_series.index)
    weights = exp_arr[~np.isnan(pp)] if weight_by_exposure else None
    return fit_trend(pp_series, times.loc[pp_series.index], kind=kind, weights=weights)
