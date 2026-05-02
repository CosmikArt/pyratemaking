"""Single-model Tweedie GLM for pure premium.

Tweedie with ``power ∈ (1, 2)`` is a compound Poisson-Gamma — a natural
single model for losses that mix a frequency mass at zero with positive
severity. ``power = 1`` is Poisson, ``power = 2`` is Gamma; ``power ≈ 1.5``
is the typical default for personal-lines auto.

Reference: Goldburd et al. 2020, ch. 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyratemaking.glm.backend import GLM, GLMResult


@dataclass
class TweedieModel:
    """Wrapper around a single Tweedie GLM with a pure-premium API."""

    fit_result: GLMResult
    power: float

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        pure_premium: pd.Series | np.ndarray,
        *,
        exposure: pd.Series | np.ndarray | None = None,
        power: float = 1.5,
        backend: str = "glum",
        alpha: float | None = None,
        l1_ratio: float = 0.0,
        base_levels: dict[str, str] | None = None,
    ) -> TweedieModel:
        glm = GLM(
            family="tweedie",
            link="log",
            backend=backend,
            tweedie_power=power,
            alpha=alpha,
            l1_ratio=l1_ratio,
            base_levels=base_levels or {},
        )
        weights = np.asarray(exposure, dtype=float) if exposure is not None else None
        result = glm.fit(X, pure_premium, sample_weight=weights)
        return cls(fit_result=result, power=power)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit_result.predict(X)

    def relativities(self, variable: str) -> pd.Series:
        return self.fit_result.relativities(variable)
