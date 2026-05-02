"""Gini coefficient and Lorenz curve.

The model Gini orders policies by predicted score and reports the
inequality of cumulative actual vs cumulative exposure. ``2 * AUC - 1``
form is used (equivalent to standard Gini up to sign).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lorenz_curve(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    *,
    weights: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Sort by predicted ascending; return cumulative weight and cumulative actual."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(p)
    a, w = a[order], w[order]
    cum_w = np.cumsum(w) / max(w.sum(), 1e-12)
    cum_a = np.cumsum(a * w) / max(np.sum(a * w), 1e-12)
    return pd.DataFrame({"cum_weight": cum_w, "cum_actual": cum_a})


def gini_coefficient(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    *,
    weights: pd.Series | np.ndarray | None = None,
    normalized: bool = False,
) -> float:
    """Model Gini coefficient.

    Parameters
    ----------
    normalized : bool, default False
        When True, divide by the perfect-model Gini to obtain the
        normalised Gini in ``[0, 1]``.
    """
    curve = lorenz_curve(actual, predicted, weights=weights)
    cum_w = curve["cum_weight"].to_numpy()
    cum_a = curve["cum_actual"].to_numpy()
    # Trapezoidal integration of cum_a vs cum_w → area under Lorenz curve.
    # Concordance-style Gini: 1 - 2 * AUC for ascending sort.
    auc = np.trapezoid(cum_a, cum_w)
    raw = 1.0 - 2.0 * auc
    if not normalized:
        return float(raw)
    perfect = lorenz_curve(actual, np.asarray(actual, dtype=float), weights=weights)
    perfect_auc = np.trapezoid(perfect["cum_actual"].to_numpy(), perfect["cum_weight"].to_numpy())
    perfect_gini = 1.0 - 2.0 * perfect_auc
    if perfect_gini == 0:
        return 0.0
    return float(raw / perfect_gini)
