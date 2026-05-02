"""Trend fitting via OLS, common for severity / frequency / pure premium."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_KIND = ("multiplicative", "exponential", "additive")


@dataclass
class Trend:
    """Result of a univariate trend fit.

    The two coefficients ``intercept`` and ``slope`` parametrise the trend
    on the *fitted* scale (log scale for multiplicative / exponential,
    raw scale for additive).
    """

    kind: str
    intercept: float
    slope: float
    slope_se: float
    times: np.ndarray
    values: np.ndarray
    residual_std: float
    n: int = field(init=False)

    def __post_init__(self) -> None:
        self.n = int(self.times.shape[0])

    @property
    def annual_change(self) -> float:
        """Annual rate of change.

        Multiplicative / exponential: ``exp(slope) - 1`` (a decimal, e.g. 0.04
        means 4% per year). Additive: ``slope`` itself (level units per year).
        """
        if self.kind in ("multiplicative", "exponential"):
            return float(np.expm1(self.slope))
        return float(self.slope)

    def predict(self, t: float | np.ndarray) -> float | np.ndarray:
        """Predict the level at time ``t``."""
        t_arr = np.asarray(t, dtype=float)
        eta = self.intercept + self.slope * t_arr
        if self.kind in ("multiplicative", "exponential"):
            return np.exp(eta)
        return eta

    def factor_to(self, t_from: float | np.ndarray, t_to: float) -> float | np.ndarray:
        """Multiplicative factor that brings level at ``t_from`` to ``t_to``."""
        return np.asarray(self.predict(t_to)) / np.asarray(self.predict(t_from))

    def confidence_interval(self, t: float, alpha: float = 0.05) -> tuple[float, float]:
        """Pointwise (1-alpha) CI for the predicted level at time ``t``.

        Uses the asymptotic OLS variance of the slope; ignores covariance
        between intercept and slope, which is a fine approximation when ``t``
        is reasonably close to ``mean(times)``.
        """
        from scipy.stats import t as student_t

        df = max(self.n - 2, 1)
        crit = student_t.ppf(1 - alpha / 2, df)
        x_bar = float(np.mean(self.times))
        Sxx = float(np.sum((self.times - x_bar) ** 2))
        if Sxx == 0:
            raise ValueError("trend times are constant; CI undefined")
        var_pred = self.residual_std**2 * (1 / self.n + (t - x_bar) ** 2 / Sxx)
        se_pred = float(np.sqrt(var_pred))
        center = self.intercept + self.slope * t
        lo, hi = center - crit * se_pred, center + crit * se_pred
        if self.kind in ("multiplicative", "exponential"):
            return float(np.exp(lo)), float(np.exp(hi))
        return float(lo), float(hi)

    def project(self, periods: list[float] | np.ndarray) -> pd.DataFrame:
        """Project predicted levels at multiple time points with CIs."""
        rows = []
        for t in np.asarray(periods, dtype=float).ravel():
            lo, hi = self.confidence_interval(t)
            rows.append(
                {
                    "period": float(t),
                    "predicted": float(self.predict(t)),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        return pd.DataFrame(rows).set_index("period")

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "kind": [self.kind],
                "annual_change": [self.annual_change],
                "slope": [self.slope],
                "slope_se": [self.slope_se],
                "n_obs": [self.n],
                "residual_std": [self.residual_std],
            }
        )

    def __repr__(self) -> str:
        return (
            f"Trend(kind={self.kind!r}, annual_change={self.annual_change:.4%}, "
            f"slope={self.slope:.6f} ± {self.slope_se:.6f}, n={self.n})"
        )


def _ols(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> tuple[float, float, float, float]:
    """Weighted least squares for ``y ~ a + b*x``. Returns ``(a, b, b_se, sigma)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    if n < 2:
        raise ValueError("need at least 2 observations to fit a trend")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    if w.sum() == 0:
        raise ValueError("all weights are zero")

    W = w / w.sum()
    x_bar = float(np.sum(W * x))
    y_bar = float(np.sum(W * y))
    Sxx = float(np.sum(W * (x - x_bar) ** 2)) * w.sum()
    Sxy = float(np.sum(W * (x - x_bar) * (y - y_bar))) * w.sum()
    if Sxx == 0:
        raise ValueError("trend times are constant; slope undefined")
    slope = Sxy / Sxx
    intercept = y_bar - slope * x_bar
    resid = y - (intercept + slope * x)
    dof = max(n - 2, 1)
    sigma = float(np.sqrt(np.sum(w * resid**2) / dof / max(np.mean(w), 1e-12)))
    slope_var = sigma**2 / Sxx
    slope_se = float(np.sqrt(max(slope_var, 0.0)))
    return intercept, slope, slope_se, sigma


def fit_trend(
    values: pd.Series | np.ndarray,
    times: pd.Series | np.ndarray | None = None,
    *,
    kind: str = "multiplicative",
    weights: np.ndarray | pd.Series | None = None,
) -> Trend:
    """Fit a univariate trend to a series of observations.

    Parameters
    ----------
    values : array-like
        Observations (e.g. average severity by AY).
    times : array-like, optional
        Time index. If ``values`` is a Series, defaults to its index.
    kind : str, default ``"multiplicative"``
        ``"multiplicative"`` and ``"exponential"`` are aliases — both fit OLS
        on ``log(values)``. ``"additive"`` fits OLS on the raw values.
    weights : array-like, optional
        Weights for the OLS fit (e.g. earned exposure).

    Returns
    -------
    Trend
    """
    if kind not in _KIND:
        raise ValueError(f"kind must be one of {_KIND}, got {kind!r}")

    if isinstance(values, pd.Series):
        if times is None:
            times = values.index.to_numpy()
        values_arr = values.to_numpy(dtype=float)
    else:
        values_arr = np.asarray(values, dtype=float)
        if times is None:
            raise ValueError("times required when values is not a Series")

    times_arr = np.asarray(times, dtype=float)
    if values_arr.shape != times_arr.shape:
        raise ValueError("values and times must have the same shape")

    if kind in ("multiplicative", "exponential"):
        if (values_arr <= 0).any():
            raise ValueError(
                "multiplicative/exponential trend requires strictly positive values"
            )
        y = np.log(values_arr)
    else:
        y = values_arr

    w_arr = None if weights is None else np.asarray(weights, dtype=float)
    intercept, slope, slope_se, sigma = _ols(times_arr, y, w_arr)
    return Trend(
        kind=kind,
        intercept=float(intercept),
        slope=float(slope),
        slope_se=float(slope_se),
        times=times_arr,
        values=values_arr,
        residual_std=float(sigma),
    )


def sensitivity_table(
    values: pd.Series | np.ndarray,
    times: pd.Series | np.ndarray | None = None,
    *,
    horizon: float,
    weights: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Compare projected level at ``horizon`` across the three trend forms.

    Useful for the trend-selection exhibit: a single table summarising
    annual change, projected level, and CI width under each candidate.
    """
    rows = []
    for kind in ("multiplicative", "additive"):
        try:
            t = fit_trend(values, times, kind=kind, weights=weights)
        except ValueError:
            continue
        lo, hi = t.confidence_interval(horizon)
        rows.append(
            {
                "kind": kind,
                "annual_change": t.annual_change,
                "projected": float(t.predict(horizon)),
                "ci_lo": lo,
                "ci_hi": hi,
                "residual_std": t.residual_std,
            }
        )
    return pd.DataFrame(rows).set_index("kind")
