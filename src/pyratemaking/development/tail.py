"""Tail factor selection for development triangles.

Four standard methods:

* :func:`bondy_tail` — Bondy (1963) assumes the next link factor equals
  the previous one (or some power of it). Closed form, no fit.
* :func:`sherman_tail` — Sherman (1984) inverse-power curve fit:
  ``LDF_j = 1 + c / j^d``.
* :func:`exponential_decay_tail` — fits ``LDF_j - 1 = a · exp(-b · j)``.
* :func:`power_curve_tail` — generic power decay ``LDF_j - 1 = a / j^b``.

Each function returns a ``(tail_factor, fit_diagnostics)`` tuple where the
diagnostics frame includes the fitted curve parameters and the projected
LDFs out to ``periods_to_extend``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _ages_from_link_factors(link_factors: pd.Series) -> np.ndarray:
    ages = []
    for label in link_factors.index:
        left, _ = str(label).split("-")
        ages.append(float(left))
    return np.asarray(ages, dtype=float)


def bondy_tail(
    link_factors: pd.Series,
    *,
    power: float = 1.0,
) -> tuple[float, pd.DataFrame]:
    """Bondy tail factor.

    The classical Bondy (1963) assumption: the tail factor equals the last
    observed link factor. ``power > 1`` extends the assumption (e.g. modified
    Bondy uses the link factor twice).
    """
    last = float(link_factors.iloc[-1])
    tail = last**power
    return tail, pd.DataFrame(
        {"method": ["bondy"], "last_link": [last], "power": [power], "tail": [tail]}
    )


def sherman_tail(
    link_factors: pd.Series,
    *,
    periods_to_extend: int = 20,
    p0: tuple[float, float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Sherman (1984) inverse-power curve fit on link factors.

    Fits ``LDF_j = 1 + c / j^d`` to the observed (age, link factor) pairs,
    then accumulates projected link factors for ``periods_to_extend`` future
    periods to obtain the tail.
    """
    ages = _ages_from_link_factors(link_factors)
    ldfs = link_factors.to_numpy(dtype=float)
    mask = (ldfs > 1) & np.isfinite(ldfs)
    if mask.sum() < 2:
        return 1.0, pd.DataFrame(
            {"method": ["sherman"], "c": [np.nan], "d": [np.nan], "tail": [1.0]}
        )

    def model(j: np.ndarray, c: float, d: float) -> np.ndarray:
        return 1.0 + c / np.power(j, d)

    p0 = p0 or (max(ldfs[mask][0] - 1, 1e-3), 1.0)
    popt, _ = curve_fit(model, ages[mask], ldfs[mask], p0=p0, maxfev=10_000)
    c, d = popt

    last_age = float(ages.max())
    proj_ages = last_age + np.arange(1, periods_to_extend + 1, dtype=float)
    proj_ldfs = model(proj_ages, c, d)
    tail = float(np.prod(proj_ldfs))
    return tail, pd.DataFrame(
        {"method": ["sherman"], "c": [float(c)], "d": [float(d)], "tail": [tail]}
    )


def exponential_decay_tail(
    link_factors: pd.Series,
    *,
    periods_to_extend: int = 20,
) -> tuple[float, pd.DataFrame]:
    """Exponential decay fit on ``LDF - 1`` against age.

    Useful when the inverse-power form does not converge.
    """
    ages = _ages_from_link_factors(link_factors)
    ldfs = link_factors.to_numpy(dtype=float)
    excess = ldfs - 1.0
    mask = (excess > 0) & np.isfinite(excess)
    if mask.sum() < 2:
        return 1.0, pd.DataFrame(
            {"method": ["exp_decay"], "a": [np.nan], "b": [np.nan], "tail": [1.0]}
        )
    log_excess = np.log(excess[mask])
    slope, intercept = np.polyfit(ages[mask], log_excess, 1)
    a = float(np.exp(intercept))
    b = float(-slope)
    last_age = float(ages.max())
    proj_ages = last_age + np.arange(1, periods_to_extend + 1, dtype=float)
    proj_ldfs = 1.0 + a * np.exp(-b * proj_ages)
    tail = float(np.prod(proj_ldfs))
    return tail, pd.DataFrame({"method": ["exp_decay"], "a": [a], "b": [b], "tail": [tail]})


def power_curve_tail(
    link_factors: pd.Series,
    *,
    periods_to_extend: int = 20,
) -> tuple[float, pd.DataFrame]:
    """Power-curve fit on ``LDF - 1 = a / j^b``.

    Equivalent to a log-log regression after subtracting 1.
    """
    ages = _ages_from_link_factors(link_factors)
    ldfs = link_factors.to_numpy(dtype=float)
    excess = ldfs - 1.0
    mask = (excess > 0) & np.isfinite(excess)
    if mask.sum() < 2:
        return 1.0, pd.DataFrame({"method": ["power"], "a": [np.nan], "b": [np.nan], "tail": [1.0]})
    log_excess = np.log(excess[mask])
    log_ages = np.log(ages[mask])
    slope, intercept = np.polyfit(log_ages, log_excess, 1)
    a = float(np.exp(intercept))
    b = float(-slope)
    last_age = float(ages.max())
    proj_ages = last_age + np.arange(1, periods_to_extend + 1, dtype=float)
    proj_ldfs = 1.0 + a / np.power(proj_ages, b)
    tail = float(np.prod(proj_ldfs))
    return tail, pd.DataFrame({"method": ["power"], "a": [a], "b": [b], "tail": [tail]})


def select_tail(
    link_factors: pd.Series,
    methods: Sequence[str] = ("bondy", "sherman", "power", "exp_decay"),
    *,
    periods_to_extend: int = 20,
) -> pd.DataFrame:
    """Run the requested tail methods and return a comparison frame."""
    out = []
    for m in methods:
        if m == "bondy":
            _, info = bondy_tail(link_factors)
        elif m == "sherman":
            _, info = sherman_tail(link_factors, periods_to_extend=periods_to_extend)
        elif m == "power":
            _, info = power_curve_tail(link_factors, periods_to_extend=periods_to_extend)
        elif m == "exp_decay":
            _, info = exponential_decay_tail(link_factors, periods_to_extend=periods_to_extend)
        else:
            raise ValueError(f"unknown tail method {m!r}")
        out.append(info)
    return pd.concat(out, ignore_index=True).set_index("method")
