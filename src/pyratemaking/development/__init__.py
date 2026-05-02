"""Loss development triangles and ultimate-loss estimation (W&M §6).

Three classic methods are exposed:

* :class:`ChainLadder` — volume-weighted age-to-age factors (Mack 1993).
* :class:`BornhuetterFerguson` — blends a priori expected losses with
  reported development.
* :class:`CapeCod` — derives the expected loss ratio from the data itself
  and feeds it into the BF formula.

Tail factors beyond the observed development span come from
:mod:`pyratemaking.development.tail` (Bondy, Sherman, exponential decay,
power curve).

When the optional ``burncost`` package is installed, :class:`ChainLadder`
also exposes :meth:`to_burncost` for bridging into that workflow.
"""

from pyratemaking.development.ldf import (
    BornhuetterFerguson,
    CapeCod,
    ChainLadder,
    age_to_age_factors,
    cumulative_factors,
)
from pyratemaking.development.tail import (
    bondy_tail,
    exponential_decay_tail,
    power_curve_tail,
    sherman_tail,
)

__all__ = [
    "BornhuetterFerguson",
    "CapeCod",
    "ChainLadder",
    "age_to_age_factors",
    "bondy_tail",
    "cumulative_factors",
    "exponential_decay_tail",
    "power_curve_tail",
    "sherman_tail",
]
