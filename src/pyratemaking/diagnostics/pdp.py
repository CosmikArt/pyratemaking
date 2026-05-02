"""Partial dependence plots for fitted GLMs / GAMs.

The partial dependence of feature ``j`` is the average prediction over the
empirical distribution of the other features at each value of ``j``. We
return a tidy frame; ALE (accumulated local effects) is exposed via
:func:`accumulated_local_effects` for unbalanced predictors where PDP can
mislead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def partial_dependence(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature: str,
    *,
    grid: Sequence[float] | None = None,
    n_grid: int = 20,
) -> pd.DataFrame:
    """Compute partial dependence for one feature.

    Parameters
    ----------
    predict_fn : callable
        Maps a DataFrame to predictions. Typically ``model.predict``.
    X : DataFrame
        Reference data; the empirical distribution of the other features.
    feature : str
        Column to vary.
    grid : sequence, optional
        Values to sweep over. Defaults to a quantile grid of the feature.
    n_grid : int, default 20
    """
    if feature not in X.columns:
        raise KeyError(f"{feature!r} not in X")
    if grid is None:
        if pd.api.types.is_numeric_dtype(X[feature]):
            quantiles = np.linspace(0.05, 0.95, n_grid)
            grid = np.quantile(X[feature].to_numpy(dtype=float), quantiles)
        else:
            grid = list(X[feature].astype("category").cat.categories)

    rows = []
    for v in grid:
        snapshot = X.copy()
        snapshot[feature] = v
        pred = predict_fn(snapshot)
        rows.append({feature: v, "pdp": float(np.mean(pred))})
    return pd.DataFrame(rows)


def accumulated_local_effects(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature: str,
    *,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Apley-Zhu (2020) ALE for a numeric feature.

    Less biased than PDP when the feature is correlated with others.
    """
    if not pd.api.types.is_numeric_dtype(X[feature]):
        raise TypeError("ALE requires a numeric feature")

    edges = np.quantile(
        X[feature].to_numpy(dtype=float), np.linspace(0, 1, n_bins + 1)
    )
    edges = np.unique(edges)  # safety against ties
    centers = (edges[:-1] + edges[1:]) / 2

    feature_values = X[feature].to_numpy(dtype=float)
    bin_idx = np.clip(np.searchsorted(edges, feature_values, side="right") - 1, 0, len(edges) - 2)

    differences = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        mask = bin_idx == k
        if not mask.any():
            continue
        lower = X.loc[mask].copy()
        upper = X.loc[mask].copy()
        lower[feature] = edges[k]
        upper[feature] = edges[k + 1]
        diff = predict_fn(upper) - predict_fn(lower)
        differences[k] = float(np.mean(diff))
        counts[k] = int(mask.sum())

    ale = np.cumsum(differences)
    ale = ale - np.mean(ale)
    return pd.DataFrame({feature: centers, "ale": ale, "n_in_bin": counts.astype(int)})
