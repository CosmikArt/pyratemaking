"""Model diagnostics for ratemaking GLMs.

Lift, A/E, Gini, residuals, partial dependence, calibration. Each function
returns either a tidy DataFrame or a matplotlib :class:`Figure` so users
can render to notebooks, save to disk, or merge into reports.
"""

from pyratemaking.diagnostics.ae import actual_vs_expected
from pyratemaking.diagnostics.calibration import reliability_diagram
from pyratemaking.diagnostics.gains import gini_coefficient, lorenz_curve
from pyratemaking.diagnostics.lift import (
    LiftChart,
    decile_analysis,
    double_lift,
    lift_table,
)
from pyratemaking.diagnostics.pdp import partial_dependence
from pyratemaking.diagnostics.residuals import deviance_residuals, pearson_residuals

__all__ = [
    "LiftChart",
    "actual_vs_expected",
    "decile_analysis",
    "deviance_residuals",
    "double_lift",
    "gini_coefficient",
    "lift_table",
    "lorenz_curve",
    "partial_dependence",
    "pearson_residuals",
    "reliability_diagram",
]
