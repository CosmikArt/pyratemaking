"""Lift charts and quantile-lift analysis.

* :func:`lift_table` — bin observations by predicted score, show observed vs
  predicted in each bin, weighted by exposure.
* :func:`double_lift` — compare two models on the same book by ranking
  policies by the *ratio* of their predictions.
* :func:`decile_analysis` — actual-to-expected by predicted decile.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def lift_table(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    *,
    weights: pd.Series | np.ndarray | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-bin observed vs predicted, sorted by predicted score (ascending).

    Returns columns ``avg_predicted``, ``avg_actual``, ``weight``, ``lift``.
    The lift is ``avg_actual / overall_avg_actual`` so a perfect model rises
    monotonically across bins.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    if a.shape != p.shape or a.shape != w.shape:
        raise ValueError("actual, predicted, weights must share shape")

    order = np.argsort(p)
    a, p, w = a[order], p[order], w[order]
    cum_w = np.cumsum(w)
    total = cum_w[-1]
    edges = np.linspace(0, total, n_bins + 1)
    rows = []
    overall_actual = float(np.sum(a * w) / max(np.sum(w), 1e-12))
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
    out["lift"] = out["avg_actual"] / max(overall_actual, 1e-12)
    return out


def double_lift(
    actual: pd.Series | np.ndarray,
    pred_a: pd.Series | np.ndarray,
    pred_b: pd.Series | np.ndarray,
    *,
    weights: pd.Series | np.ndarray | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Sort policies by ``pred_a / pred_b`` and report observed averages.

    A model that is genuinely better will show observed averages tracking
    its predictions across the sort.
    """
    a = np.asarray(actual, dtype=float)
    pa = np.asarray(pred_a, dtype=float)
    pb = np.asarray(pred_b, dtype=float)
    if (pb <= 0).any():
        raise ValueError("pred_b must be strictly positive for the ratio sort")
    ratio = pa / pb
    order = np.argsort(ratio)
    a = a[order]
    pa = pa[order]
    pb = pb[order]
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)[order]
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
                "avg_actual": float(np.sum(a[mask] * w[mask]) / max(wgt, 1e-12)),
                "avg_pred_a": float(np.sum(pa[mask] * w[mask]) / max(wgt, 1e-12)),
                "avg_pred_b": float(np.sum(pb[mask] * w[mask]) / max(wgt, 1e-12)),
            }
        )
    return pd.DataFrame(rows).set_index("bin")


def decile_analysis(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    *,
    weights: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Actual-to-expected by predicted decile."""
    table = lift_table(actual, predicted, weights=weights, n_bins=10)
    table["a_to_e"] = table["avg_actual"] / table["avg_predicted"].where(table["avg_predicted"] > 0)
    return table


@dataclass
class LiftChart:
    """Render a lift table as a matplotlib chart."""

    table: pd.DataFrame

    def figure(self, title: str | None = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = self.table.index
        ax.plot(x, self.table["avg_actual"], "o-", label="Actual")
        ax.plot(x, self.table["avg_predicted"], "s--", label="Predicted")
        ax.set_xlabel("Predicted score bin")
        ax.set_ylabel("Average value")
        ax.set_title(title or "Lift chart")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def show(self) -> matplotlib.figure.Figure:  # pragma: no cover - manual
        fig = self.figure()
        plt.show()
        return fig
