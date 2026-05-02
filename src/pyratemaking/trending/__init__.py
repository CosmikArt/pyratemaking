"""Loss and premium trending (W&M §7).

The trending workflow takes a per-period series — average severity, frequency,
pure premium, average premium — fits a parametric trend, and projects forward
to the rating-effective period. We expose three functional forms:

* **Multiplicative** / **exponential** — OLS on ``log(y)`` against time. Rate
  of change is constant and reported as an annual percentage.
* **Additive** — OLS on ``y`` against time. Rate of change is in level units
  per year (e.g., +$50/year for severity).

Confidence intervals come from the OLS standard error of the slope. The fitted
:class:`Trend` exposes :meth:`Trend.project` and :meth:`Trend.factor_to` so
callers can both see projected values and apply a trend factor to historical
points.
"""

from pyratemaking.trending.severity import severity_trend
from pyratemaking.trending.frequency import frequency_trend
from pyratemaking.trending.pure_premium import pure_premium_trend
from pyratemaking.trending.core import Trend, fit_trend

__all__ = [
    "Trend",
    "fit_trend",
    "frequency_trend",
    "pure_premium_trend",
    "severity_trend",
]
