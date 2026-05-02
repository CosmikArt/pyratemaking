"""Multi-way classification analysis (W&M §§9–10).

Wraps a Tweedie or frequency-severity GLM, extracts relativities for the
categorical rating variables, calibrates a scale factor so the rated
premium matches a target, and exposes a simple ``predict_premium`` API.
Numeric rating variables (e.g. driver age, sum insured) participate in the
GLM directly — they don't show up in the relativity tables but their
effect is fully captured in :meth:`ClassificationResult.predict_premium`.
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
from pyratemaking.relativities.multi_way import multi_way_relativities


@dataclass
class ClassificationResult:
    """Output of a classification analysis."""

    rating_vars: list[str]
    backend: str
    family: str
    relativities: dict[str, pd.Series]
    base_rate: float
    scale_factor: float
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

    def predict_premium(
        self,
        policies: pd.DataFrame,
        exposure_col: str = "exposure",
    ) -> np.ndarray:
        """Predicted total premium per policy (loss + load implicit in scaling)."""
        x = policies[self.rating_vars]
        pp_per_unit = self._predict_pure_premium(x)
        exposure = policies[exposure_col].to_numpy(dtype=float)
        return self.scale_factor * pp_per_unit * exposure

    def _predict_pure_premium(self, x: pd.DataFrame) -> np.ndarray:
        if self.family == "tweedie":
            tw: TweedieModel = self.raw_model  # type: ignore[assignment]
            return np.asarray(tw.predict(x), dtype=float)
        if self.family == "frequency_severity":
            fs: FrequencySeverityModel = self.raw_model  # type: ignore[assignment]
            return np.asarray(fs.predict(x, exposure=np.ones(len(x))), dtype=float)
        raise ValueError(f"unsupported family for prediction: {self.family!r}")

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ("rating_vars", ", ".join(self.rating_vars)),
                ("backend", self.backend),
                ("family", self.family),
                ("base_rate", f"{self.base_rate:.4f}"),
                ("scale_factor", f"{self.scale_factor:.6f}"),
                ("off_balance", f"{self.off_balance:.6f}"),
            ],
            columns=["item", "value"],
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

    See module docstring for the contract.
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
        cat_vars = [v for v in rating_vars if not pd.api.types.is_numeric_dtype(X[v])]
        rels = multi_way_relativities(glm, cat_vars)
        raw_model: object = tw
    elif family == "frequency_severity":
        if count_col is None or count_col not in policies.columns:
            raise KeyError("frequency_severity classification requires count_col")
        fs = FrequencySeverityModel.fit(
            X,
            policies[count_col],
            policies[loss_col],
            policies[exposure_col],
            backend=backend,
            base_levels=base_levels,
        )
        cat_vars = [v for v in rating_vars if not pd.api.types.is_numeric_dtype(X[v])]
        rels = {
            v: fs.frequency.relativities(v) * fs.severity.relativities(v)
            for v in cat_vars
        }
        raw_model = fs
    else:
        raise ValueError(
            f"family must be 'tweedie' or 'frequency_severity', got {family!r}"
        )

    pp_pred = _predict_pp(raw_model, family, X)
    base_rate = float(np.mean(pp_pred))

    raw_total_premium = float((pp_pred * exposure).sum())
    if raw_total_premium <= 0:
        raise ValueError("predicted pure premium is zero; cannot calibrate")
    if target_average_premium is None:
        target_total = float(losses.sum())
    else:
        target_total = float(target_average_premium * exposure.sum())

    scale_factor = target_total / raw_total_premium
    rated_total = scale_factor * raw_total_premium
    off_balance = float(losses.sum() / rated_total) if rated_total > 0 else float("nan")

    return ClassificationResult(
        rating_vars=rating_vars,
        backend=backend,
        family=family,
        relativities=rels,
        base_rate=base_rate * scale_factor,
        scale_factor=scale_factor,
        off_balance=off_balance,
        raw_model=raw_model,
    )


def _predict_pp(model: object, family: str, X: pd.DataFrame) -> np.ndarray:
    if family == "tweedie":
        tw: TweedieModel = model  # type: ignore[assignment]
        return np.asarray(tw.predict(X), dtype=float)
    if family == "frequency_severity":
        fs: FrequencySeverityModel = model  # type: ignore[assignment]
        return np.asarray(fs.predict(X, exposure=np.ones(len(X))), dtype=float)
    raise ValueError(family)
