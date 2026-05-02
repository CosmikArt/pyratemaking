"""Parallelogram method (W&M §5.2).

The cumulative rate level is a step function of policy effective date. For
annual policies written uniformly, calendar-year ``Y`` earns 50% of its premium
from policies effective in ``Y-1`` and 50% from policies effective in ``Y``.
The earned-exposure density across effective dates is a triangular hat with
peak at ``u = Y`` and support ``[Y-1, Y+1]``.

The average rate level for CY ``Y`` is the integral of the rate-level step
function against this hat. We compute it in closed form per piecewise-constant
segment to avoid quadrature noise — handy when reproducing textbook tables.

Limitations of the geometric method (CAS practice notes): uniform writing,
identical policy term across the history, no mid-term endorsements. For
non-uniform writing or mixed terms, use :func:`extension_of_exposures`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RateChange:
    """A single multiplicative rate change.

    Parameters
    ----------
    date : float
        Effective date as a fractional year (e.g. ``2011.25`` for 1 April 2011).
        Pass :func:`to_fractional_year` if you have a real date.
    factor : float
        Multiplicative factor applied to the prior rate level.
        ``1.05`` represents a +5% rate increase.
    """

    date: float
    factor: float


def to_fractional_year(date: pd.Timestamp | str) -> float:
    """Convert a date to fractional-year form (e.g. 2011-04-01 → 2011.25).

    Uses the proportion of the calendar year already elapsed at midnight
    of ``date``. Leap years use 366-day denominators.
    """
    ts = pd.Timestamp(date)
    year_start = pd.Timestamp(year=ts.year, month=1, day=1)
    next_year_start = pd.Timestamp(year=ts.year + 1, month=1, day=1)
    elapsed = (ts - year_start).total_seconds()
    full = (next_year_start - year_start).total_seconds()
    return ts.year + elapsed / full


def _segments(
    rate_changes: Sequence[RateChange],
    starting_factor: float,
    span: tuple[float, float],
) -> list[tuple[float, float, float]]:
    """Return ``[(start, end, cumulative_factor), ...]`` covering ``span``.

    ``starting_factor`` is the cumulative rate level in effect at ``span[0]``.
    Changes outside ``span`` clip the boundary segments.
    """
    lo, hi = span
    sorted_changes = sorted(rate_changes, key=lambda c: c.date)

    boundaries = [lo]
    factors_after = []
    cumulative = starting_factor
    for ch in sorted_changes:
        if ch.date <= lo:
            cumulative *= ch.factor
            continue
        if ch.date >= hi:
            break
        boundaries.append(ch.date)
        factors_after.append(cumulative)
        cumulative *= ch.factor
    boundaries.append(hi)
    factors_after.append(cumulative)

    return [(boundaries[i], boundaries[i + 1], factors_after[i]) for i in range(len(factors_after))]


def _hat_integral(a: float, b: float, year: int) -> float:
    """Integral of the triangular hat ``w(u; year)`` over ``[a, b]``.

    The hat rises linearly from 0 at ``year - 1`` to 1 at ``year`` and falls
    back to 0 at ``year + 1``. Total area is 1.
    """
    if b <= a:
        return 0.0

    def left_part(x: float) -> float:
        x_clip = min(max(x, year - 1), year)
        return (x_clip - (year - 1)) ** 2 / 2

    def right_part(x: float) -> float:
        x_clip = min(max(x, year), year + 1)
        return -(((year + 1) - x_clip) ** 2) / 2

    return (left_part(b) - left_part(a)) + (right_part(b) - right_part(a))


def average_rate_level(
    year: int,
    rate_changes: Iterable[RateChange],
    starting_factor: float = 1.0,
) -> float:
    """Average rate level in CY ``year`` under the parallelogram method.

    Parameters
    ----------
    year : int
        Calendar year.
    rate_changes : iterable of RateChange
        All rate changes ever applied. Out-of-range changes are absorbed
        into the starting level or ignored.
    starting_factor : float, default 1.0
        Cumulative factor at ``year - 1`` (the left edge of the hat).

    Returns
    -------
    float
        Earned-exposure-weighted average cumulative rate factor for the year.
    """
    changes = list(rate_changes)
    span = (year - 1.0, year + 1.0)
    segments = _segments(changes, starting_factor, span)
    return float(sum(f * _hat_integral(a, b, year) for a, b, f in segments))


def parallelogram(
    earned_premium_by_ay: pd.Series,
    rate_changes: Iterable[RateChange],
    *,
    starting_factor: float = 1.0,
    current_factor: float | None = None,
) -> pd.DataFrame:
    """Apply the parallelogram method to a vector of earned premium by AY.

    Parameters
    ----------
    earned_premium_by_ay : Series
        Indexed by integer AY, values are collected/earned premium.
    rate_changes : iterable of RateChange
    starting_factor : float, default 1.0
        Rate level in effect at the start of the earliest AY in the index.
    current_factor : float, optional
        Most recent cumulative factor. Defaults to the product of all
        provided rate changes applied to ``starting_factor``.

    Returns
    -------
    DataFrame
        Indexed by AY, columns ``earned_premium``, ``avg_rate_level``,
        ``on_level_factor``, ``on_level_premium``.
    """
    changes = list(rate_changes)
    if current_factor is None:
        current_factor = starting_factor
        for ch in changes:
            current_factor *= ch.factor

    rows = []
    for ay in earned_premium_by_ay.index:
        avg = average_rate_level(int(ay), changes, starting_factor)
        olf = current_factor / avg
        prem = float(earned_premium_by_ay.loc[ay])
        rows.append(
            {
                "ay": int(ay),
                "earned_premium": prem,
                "avg_rate_level": avg,
                "on_level_factor": olf,
                "on_level_premium": prem * olf,
            }
        )
    return pd.DataFrame(rows).set_index("ay")


def on_level_factors(
    years: Sequence[int],
    rate_changes: Iterable[RateChange],
    *,
    starting_factor: float = 1.0,
    current_factor: float | None = None,
) -> pd.Series:
    """Return on-level factors only, without bringing premium along."""
    changes = list(rate_changes)
    if current_factor is None:
        current_factor = starting_factor
        for ch in changes:
            current_factor *= ch.factor
    avgs = np.array([average_rate_level(int(y), changes, starting_factor) for y in years])
    return pd.Series(current_factor / avgs, index=list(years), name="on_level_factor")
