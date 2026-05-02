"""GLMs for ratemaking with two interchangeable backends.

Two engines sit behind one API:

* ``backend="glum"`` — fast, modern, Tweedie-native (default).
* ``backend="statsmodels"`` — classic, exam-comparable output, full inference.

Same parameter names (``family``, ``link``, ``exposure``, ``sample_weight``)
across both. The fitted result exposes named coefficients, predictions, and
deviance / log-likelihood.
"""

from pyratemaking.glm.backend import GLM, GLMResult
from pyratemaking.glm.constrained import (
    fit_monotone_glm,
    monotone_relativities,
)
from pyratemaking.glm.families import family_spec
from pyratemaking.glm.frequency_severity import (
    FrequencySeverityModel,
    fit_frequency,
    fit_severity,
)
from pyratemaking.glm.penalized import fit_penalized
from pyratemaking.glm.stepwise import stepwise_select
from pyratemaking.glm.tweedie import TweedieModel

__all__ = [
    "GLM",
    "FrequencySeverityModel",
    "GLMResult",
    "TweedieModel",
    "family_spec",
    "fit_frequency",
    "fit_monotone_glm",
    "fit_penalized",
    "fit_severity",
    "monotone_relativities",
    "stepwise_select",
]
