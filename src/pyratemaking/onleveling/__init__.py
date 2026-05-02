"""Bring historical premium to current rate level.

Two methods are exposed:

* :func:`parallelogram` — geometric approximation under uniform writing
  (W&M §5.2). Cheap, schedule-only.
* :func:`extension_of_exposures` — re-rate each historical record using
  the current rating algorithm (W&M §5.3). Exact, requires policy detail.
"""

from pyratemaking.onleveling.extension_exposures import (
    extension_of_exposures,
    rate_under_algorithm,
)
from pyratemaking.onleveling.parallelogram import (
    RateChange,
    average_rate_level,
    on_level_factors,
    parallelogram,
)

__all__ = [
    "RateChange",
    "average_rate_level",
    "extension_of_exposures",
    "on_level_factors",
    "parallelogram",
    "rate_under_algorithm",
]
