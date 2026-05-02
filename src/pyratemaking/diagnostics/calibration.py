"""Reliability diagrams and calibration-by-segment.

Reliability diagrams compare observed averages against predicted averages
in equal-quantile bins. Well-calibrated models track the diagonal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_diagram(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
    weights: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a reliability table by predicted-quantile bin."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)

    order = np.argsort(p)
    a, p, w = a[order], p[order], w[order]
    cum_w = np.cumsum(w)
    edges = np.linspace(0, cum_w[-1], n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (cum_w > lo) & (cum_w <= hi + 1e-12)
        if not mask.any():
            continue
        wgt = float(w[mask].sum())
        rows.append(
            {
                "bin": i + 1,
                "weight": wgt,
                "avg_predicted": float(np.sum(p[mask] * w[mask]) / max(wgt, 1e-12)),
                "avg_actual": float(np.sum(a[mask] * w[mask]) / max(wgt, 1e-12)),
            }
        )
    out = pd.DataFrame(rows).set_index("bin")
    out["bias"] = out["avg_actual"] - out["avg_predicted"]
    return out
