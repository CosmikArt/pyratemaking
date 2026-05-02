"""Multi-way classification analysis (W&M §§9–10).

Wraps a Tweedie or frequency-severity GLM, extracts relativities, applies the
balance principle (W&M §10.4), computes the off-balance correction, and
calibrates a base rate consistent with the indicated total premium.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pyratemaking.glm import (
    FrequencySeverityModel,
    GLMResult,
    TweedieModel,
)
from pyratemaking.relativities.multi_way import (
    balance_principle_check,
    multi_way_relativities,
)


@dataclass
class ClassificationResult:
    """Output of a classification analysis."""

    rating_vars: list[str]
    backend: str
    family: str
    relativities: dict[str, pd.Series]
    base_rate: float
    off_balance: float
    raw_model: object = field(default=None, repr=False)

    def relativities_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for var, rel in self.relativities.items():
            for level, value in rel.items():
                rows.append(
                    {
                        "variable": var,
                        "level": level,
                        "relativity": float(value),
                    }
                )
        return pd.DataFrame(rows)

    def predict_premium(self, policies: pd.DataFrame, exposure_col: str = "exposure") -> np.ndarray:
        factors = np.ones(len(policies))
        for var, rel in self.relativities.items():
            factors *= policies[var].map(rel.astype(float).to_dict()).to_numpy(dtype=float)
        return self.base_rate * factors * policies[exposure_col].to_numpy(dtype=float)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "item": [
                    "rating_vars",
                    "backend",
                    "family",
                    "base_rate",
                    "off_balance",
                ],
                "value": [
                    ", ".join(self.rating_vars),
                    self.backend,
                    self.family,
                    f"{self.base_rate:.4f}",
                    f"{self.off_balance:.4f}",
                ],
            }
        ).set_index("item")

    def __repr__(self) -> str:
        return (
            f"ClassificationResult(family={self.family!r}, vars={self.rating_vars}, "
            f"base_rate={self.base_rate:.2f}, off_balance={self.off_balance:.4f})"
        )


def classify(
    policies: pd.DataFrame,
    *,
    rating_vars: Sequence[str],
    family: str = "tweedie",
    backend: str = "glum",
    tweedie_power: float = 1.5,
    exposure_col: str = "exposure",
    loss_col: str = "incurred_losses",
    count_col: str | None = None,
    base_levels: dict[str, str] | None = None,
    target_average_premium: float | None = None,
) -> ClassificationResult:
    """Run a multi-way classification analysis end to end.

    Parameters
    ----------
    policies : DataFrame
        One row per earned-exposure period with rating variables, exposure,
        and incurred losses.
    rating_vars : sequence of str
        Variables to fit. Categorical and numeric both supported.
    family : str
        ``"tweedie"`` for a single GLM on pure premium (default), or
        ``"frequency_severity"`` for a Poisson + Gamma pair (requires
        ``count_col``).
    backend : str
        ``"glum"`` (default) or ``"statsmodels"``.
    tweedie_power : float
        Power for Tweedie family. Ignored for frequency-severity.
    target_average_premium : float, optional
        If provided, the base rate is calibrated so that the average rated
        premium equals this target. Otherwise it is calibrated to the average
        observed loss + an implicit zero load.

    Returns
    -------
    :class:`ClassificationResult`.
    """
    rating_vars = list(rating_vars)
    base_levels = dict(base_levels or {})
    X = policies[rating_vars]
    losses = policies[loss_col].to_numpy(dtype=float)
    exposure = policies[exposure_col].to_numpy(dtype=float)

    if family == "tweedie":
        pp = np.where(exposure > 0, losses / np.where(exposure > 0, exposure, 1.0), 0.0)
        tw = TweedieModel.fit(
            X,
            pp,
            exposure=exposure,
            power=tweedie_power,
            backend=backend,
            base_levels=base_levels,
        )
        glm: GLMResult = tw.fit_result
        rels = multi_way_relativities(glm, rating_vars)
        raw_model: object = tw
    elif family == "frequency_severity":
        if count_col is None or count_col not in policies.columns:
            raise KeyError(
                "frequency_severity classification requires count_col"
            )
        fs = FrequencySeverityModel.fit(
            X,
            policies[count_col],
            policies[loss_col],
            policies[exposure_col],
            backend=backend,
            base_levels=base_levels,
        )
        rels = {
            v: fs.frequency.relativities(v) * fs.severity.relativities(v)
            for v in rating_vars
            if not np.issubdtype(X[v].dtype, np.number)
        }
        raw_model = fs
    else:
        raise ValueError(
            f"family must be 'tweedie' or 'frequency_severity', got {family!r}"
        )

    base_rate = _calibrate_base_rate(
        policies,
        rels,
        rating_vars,
        exposure_col=exposure_col,
        loss_col=loss_col,
        target_average_premium=target_average_premium,
    )
    bp = balance_principle_check(
        policies,
        rels,
        base_rate,
        exposure_col=exposure_col,
        loss_col=loss_col,
    )
    return ClassificationResult(
        rating_vars=rating_vars,
        backend=backend,
        family=family,
        relativities=rels,
        base_rate=base_rate,
        off_balance=float(bp["off_balance"]),
        raw_model=raw_model,
    )


def _calibrate_base_rate(
    policies: pd.DataFrame,
    relativities: dict[str, pd.Series],
    rating_vars: Sequence[str],
    *,
    exposure_col: str,
    loss_col: str,
    target_average_premium: float | None,
) -> float:
    factors = np.ones(len(policies))
    for v in rating_vars:
        if v not in relativities:
            continue
        rel_map = relativities[v].astype(float).to_dict()
        factors *= policies[v].map(rel_map).to_numpy(dtype=float)
    exposure = policies[exposure_col].to_numpy(dtype=float)
    weighted_factors = (factors * exposure).sum()
    if weighted_factors <= 0:
        raise ValueError("relativities collapse to zero; cannot calibrate base rate")
    if target_average_premium is None:
        total = float(policies[loss_col].sum())
        return total / weighted_factors
    avg_factor = weighted_factors / exposure.sum()
    return float(target_average_premium / avg_factor)
