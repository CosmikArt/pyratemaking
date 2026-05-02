"""Family-aware residuals.

Pearson and deviance residuals for the families that ship in
:mod:`pyratemaking.glm`. Useful for residual plots and outlier checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pearson_residuals(
    y: pd.Series | np.ndarray,
    mu: pd.Series | np.ndarray,
    *,
    family: str,
    weights: pd.Series | np.ndarray | None = None,
    tweedie_power: float = 1.5,
) -> np.ndarray:
    """Pearson residuals: ``(y - mu) / sqrt(V(mu))`` weighted."""
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    w = np.ones_like(y_arr) if weights is None else np.asarray(weights, dtype=float)
    var = _variance(mu_arr, family, tweedie_power)
    return np.sqrt(w) * (y_arr - mu_arr) / np.sqrt(np.where(var > 0, var, 1e-12))


def deviance_residuals(
    y: pd.Series | np.ndarray,
    mu: pd.Series | np.ndarray,
    *,
    family: str,
    weights: pd.Series | np.ndarray | None = None,
    tweedie_power: float = 1.5,
) -> np.ndarray:
    """Signed deviance residuals."""
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    w = np.ones_like(y_arr) if weights is None else np.asarray(weights, dtype=float)
    sign = np.sign(y_arr - mu_arr)
    family = family.lower()
    if family == "poisson":
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(
                y_arr > 0, y_arr * np.log(y_arr / np.where(mu_arr > 0, mu_arr, 1e-12)), 0.0
            )
        d = 2 * (term - (y_arr - mu_arr))
    elif family == "gamma":
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y_arr > 0, np.log(y_arr / np.where(mu_arr > 0, mu_arr, 1e-12)), 0.0)
        d = 2 * (-term + (y_arr - mu_arr) / np.where(mu_arr > 0, mu_arr, 1e-12))
    elif family == "tweedie":
        p = tweedie_power
        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = np.where(
                y_arr > 0, y_arr ** (2 - p) / ((1 - p) * (2 - p)), 0.0
            ) - y_arr * mu_arr ** (1 - p) / (1 - p)
            term2 = mu_arr ** (2 - p) / (2 - p)
        d = 2 * (term1 + term2)
    elif family == "gaussian":
        d = (y_arr - mu_arr) ** 2
    else:
        raise ValueError(f"unsupported family for deviance residuals: {family!r}")
    return sign * np.sqrt(np.maximum(w * d, 0.0))


def _variance(mu: np.ndarray, family: str, tweedie_power: float) -> np.ndarray:
    family = family.lower()
    if family == "poisson":
        return mu
    if family == "gamma":
        return mu**2
    if family == "tweedie":
        return mu**tweedie_power
    if family == "gaussian":
        return np.ones_like(mu)
    if family in ("inverse_gaussian", "ig"):
        return mu**3
    raise ValueError(f"unknown family for variance: {family!r}")
