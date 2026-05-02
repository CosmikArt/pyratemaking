"""Frequency-severity GLM pair.

Two GLMs share the same rating-variable design matrix:

* **Frequency** — Poisson on claim counts with ``log(exposure)`` offset.
* **Severity** — Gamma on average claim amount, weighted by claim count.

Pure-premium prediction is the product of the two means. Equivalent under
mild assumptions to a single Tweedie GLM with ``power ∈ (1, 2)`` (W&M §12).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyratemaking.glm.backend import GLM, GLMResult


def fit_frequency(
    X: pd.DataFrame,
    claim_count: pd.Series | np.ndarray,
    exposure: pd.Series | np.ndarray,
    *,
    backend: str = "glum",
    base_levels: dict[str, str] | None = None,
) -> GLMResult:
    """Poisson frequency GLM with log-exposure offset."""
    glm = GLM(family="poisson", link="log", backend=backend, base_levels=base_levels or {})
    return glm.fit(X, claim_count, exposure=exposure)


def fit_severity(
    X: pd.DataFrame,
    claim_amount: pd.Series | np.ndarray,
    claim_count: pd.Series | np.ndarray,
    *,
    backend: str = "glum",
    base_levels: dict[str, str] | None = None,
) -> GLMResult:
    """Gamma severity GLM weighted by claim count.

    ``claim_amount`` is the *total* incurred loss for the row; we divide by
    ``claim_count`` to obtain the average severity that the GLM models.
    Rows with zero claims are dropped.
    """
    counts = np.asarray(claim_count, dtype=float)
    losses = np.asarray(claim_amount, dtype=float)
    mask = counts > 0
    if not mask.any():
        raise ValueError("severity model requires at least one row with claims")
    avg_sev = np.where(mask, losses / np.where(mask, counts, 1.0), np.nan)

    glm = GLM(family="gamma", link="log", backend=backend, base_levels=base_levels or {})
    return glm.fit(
        X.loc[mask].reset_index(drop=True),
        avg_sev[mask],
        sample_weight=counts[mask],
    )


@dataclass
class FrequencySeverityModel:
    """Composed frequency × severity model."""

    frequency: GLMResult
    severity: GLMResult

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        claim_count: pd.Series | np.ndarray,
        claim_amount: pd.Series | np.ndarray,
        exposure: pd.Series | np.ndarray,
        *,
        backend: str = "glum",
        base_levels: dict[str, str] | None = None,
    ) -> FrequencySeverityModel:
        freq = fit_frequency(X, claim_count, exposure, backend=backend, base_levels=base_levels)
        sev = fit_severity(X, claim_amount, claim_count, backend=backend, base_levels=base_levels)
        return cls(frequency=freq, severity=sev)

    def predict(
        self,
        X: pd.DataFrame,
        exposure: pd.Series | np.ndarray | None = None,
    ) -> np.ndarray:
        """Predicted pure premium per row.

        ``predicted_count_per_unit_exposure × predicted_severity``. If
        ``exposure`` is provided, the count contribution is scaled.
        """
        if exposure is None:
            exposure = np.ones(len(X))
        freq_mean = self.frequency.predict(X, exposure=exposure)
        sev_mean = self.severity.predict(X)
        return freq_mean * sev_mean

    def relativities(self, variable: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frequency": self.frequency.relativities(variable),
                "severity": self.severity.relativities(variable),
            }
        ).assign(pure_premium=lambda d: d["frequency"] * d["severity"])
