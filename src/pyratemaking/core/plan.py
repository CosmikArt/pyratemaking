"""End-to-end :class:`RatePlan` orchestrator.

Composes IO, indication, classification, implementation, diagnostics, and
reporting into one object so the typical workflow is a few method calls.
The README quickstart matches this API exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pyratemaking._version import __version__
from pyratemaking.core.classification import ClassificationResult, classify
from pyratemaking.core.implementation import (
    ImplementationResult,
    implement_rate_change,
)
from pyratemaking.core.indication import (
    ExpenseProvision,
    Indication,
    loss_ratio_indication,
    pure_premium_indication,
)
from pyratemaking.diagnostics import (
    LiftChart,
    actual_vs_expected,
    decile_analysis,
    gini_coefficient,
    lift_table,
)
from pyratemaking.io.claims import (
    ClaimsSchema,
    aggregate_to_ay as agg_claims,
    merge_policy_losses,
    validate_claims,
)
from pyratemaking.io.policies import (
    PolicySchema,
    aggregate_to_ay as agg_policies,
    validate_policies,
)
from pyratemaking.reporting import RatePlanReport, write_excel


@dataclass
class _Diagnostics:
    """Diagnostic helpers bound to a fitted :class:`RatePlan`."""

    plan: RatePlan

    def lift(self, n_bins: int = 10) -> LiftChart:
        actual, predicted, weight = self.plan._aligned_actual_predicted()
        return LiftChart(lift_table(actual, predicted, weights=weight, n_bins=n_bins))

    def decile(self) -> pd.DataFrame:
        actual, predicted, weight = self.plan._aligned_actual_predicted()
        return decile_analysis(actual, predicted, weights=weight)

    def gini(self, normalized: bool = False) -> float:
        actual, predicted, weight = self.plan._aligned_actual_predicted()
        return gini_coefficient(actual, predicted, weights=weight, normalized=normalized)

    def actual_vs_expected(self, by: str | Sequence[str] | None = None) -> pd.DataFrame:
        df = self.plan._with_predictions()
        return actual_vs_expected(
            df,
            actual_col="incurred_losses",
            expected_col="expected_premium",
            by=by,
            weight_col=None,
        )


@dataclass
class _ReportFacade:
    """Reporting helpers bound to a :class:`RatePlan`."""

    plan: RatePlan

    def filing(self, path: str | Path) -> Path:
        report = self.plan._build_filing_report()
        return report.write(path)

    def excel(self, path: str | Path) -> Path:
        sheets = self.plan._excel_sheets()
        return write_excel(path, sheets)

    def report(self) -> RatePlanReport:
        return self.plan._build_filing_report()


@dataclass
class RatePlan:
    """End-to-end ratemaking workflow object.

    Parameters
    ----------
    policies : DataFrame
        Per-policy table with at least an exposure column and an AY column.
    claims : DataFrame
        Per-claim table with a loss column and an AY column.
    exposure_col, loss_col, ay_col : str
        Column names. Defaults match :class:`pyratemaking.io.policies.PolicySchema`.
    """

    policies: pd.DataFrame
    claims: pd.DataFrame
    exposure_col: str = "exposure"
    loss_col: str = "claim_amount"
    ay_col: str = "policy_ay"
    premium_col: str | None = "earned_premium"

    indication_: Indication | None = field(default=None, init=False)
    classification: ClassificationResult | None = field(default=None, init=False)
    implementation: ImplementationResult | None = field(default=None, init=False)
    diagnostics: _Diagnostics = field(init=False)
    report: _ReportFacade = field(init=False)

    def __post_init__(self) -> None:
        self._policy_schema = PolicySchema(
            exposure=self.exposure_col,
            ay=self.ay_col,
            premium=self.premium_col,
        )
        self._claims_schema = ClaimsSchema(loss=self.loss_col, ay=self.ay_col)
        self.policies = validate_policies(self.policies, self._policy_schema)
        self.claims = validate_claims(self.claims, self._claims_schema)
        self._policies_with_losses = merge_policy_losses(
            self.policies, self.claims, claims_schema=self._claims_schema
        )
        self.diagnostics = _Diagnostics(self)
        self.report = _ReportFacade(self)

    # ----- step 1 --------------------------------------------------------
    def indicate(
        self,
        *,
        method: str = "loss_ratio",
        target_lr: float | None = None,
        expenses: ExpenseProvision | None = None,
        credibility: float = 1.0,
        complement: float = 0.0,
        current_average_rate: float | None = None,
        fixed_expense_per_exposure: float = 0.0,
    ) -> Indication:
        """Run the overall rate-level indication."""
        expenses = expenses or ExpenseProvision()
        ay_pol = agg_policies(self.policies, self._policy_schema)
        ay_clm = agg_claims(self.claims, self._claims_schema)
        joined = ay_pol.join(ay_clm, how="left").fillna(0.0)

        if method == "loss_ratio":
            premium = joined["earned_premium"].sum() if "earned_premium" in joined.columns else 0.0
            if premium <= 0:
                raise ValueError(
                    "loss-ratio method requires positive earned_premium; "
                    "either supply a premium column or use method='pure_premium'"
                )
            losses = joined["incurred_losses"].sum()
            self.indication_ = loss_ratio_indication(
                on_level_premium=float(premium),
                ultimate_losses=float(losses),
                expenses=expenses,
                target_loss_ratio=target_lr,
                credibility=credibility,
                complement=complement,
            )
        elif method == "pure_premium":
            if current_average_rate is None:
                raise ValueError("pure_premium method requires current_average_rate")
            self.indication_ = pure_premium_indication(
                earned_exposure=float(joined["exposure"].sum()),
                ultimate_losses=float(joined["incurred_losses"].sum()),
                expenses=expenses,
                fixed_expense_per_exposure=fixed_expense_per_exposure,
                current_average_rate=current_average_rate,
                credibility=credibility,
                complement=complement,
            )
        else:
            raise ValueError(f"method must be 'loss_ratio' or 'pure_premium', got {method!r}")
        return self.indication_

    # ----- step 2 --------------------------------------------------------
    def classify(
        self,
        *,
        rating_vars: Sequence[str],
        family: str = "tweedie",
        backend: str = "glum",
        power: float = 1.5,
        base_levels: dict[str, str] | None = None,
        penalty: str | None = None,  # accepted for API parity; ignored on tweedie path
        alpha: float | str | None = None,
        target_average_premium: float | None = None,
    ) -> ClassificationResult:
        """Fit the classification GLM and derive relativities."""
        df = self._policies_with_losses
        self.classification = classify(
            df,
            rating_vars=rating_vars,
            family=family,
            backend=backend,
            tweedie_power=power,
            exposure_col=self.exposure_col,
            loss_col="incurred_losses",
            count_col="claim_count" if "claim_count" in df.columns else None,
            base_levels=base_levels,
            target_average_premium=target_average_premium,
        )
        return self.classification

    # ----- step 3 --------------------------------------------------------
    def implement(
        self,
        *,
        cap: float | None = None,
        floor: float | None = None,
        current_premium_col: str | None = None,
    ) -> ImplementationResult:
        """Apply caps and floors and produce the impact frame."""
        if self.classification is None:
            raise RuntimeError("call .classify(...) before .implement(...)")
        col = current_premium_col or self.premium_col or "earned_premium"
        if col not in self.policies.columns:
            raise KeyError(
                f"current premium column {col!r} not in policies; "
                f"pass current_premium_col=..."
            )
        indicated = self.classification.predict_premium(
            self.policies, exposure_col=self.exposure_col
        )
        impact_df = self.policies.copy()
        impact_df["indicated_premium"] = indicated
        self.implementation = implement_rate_change(
            impact_df,
            current_premium_col=col,
            indicated_premium_col="indicated_premium",
            cap=cap,
            floor=floor,
            extra_columns=[
                v
                for v in (
                    self.classification.rating_vars
                    if self.classification is not None
                    else []
                )
                if v in self.policies.columns
            ],
        )
        return self.implementation

    # ----- helpers -------------------------------------------------------
    def _with_predictions(self) -> pd.DataFrame:
        if self.classification is None:
            raise RuntimeError("classify() must run before this diagnostic")
        df = self._policies_with_losses.copy()
        df["expected_premium"] = self.classification.predict_premium(
            df, exposure_col=self.exposure_col
        )
        return df

    def _aligned_actual_predicted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = self._with_predictions()
        return (
            df["incurred_losses"].to_numpy(dtype=float),
            df["expected_premium"].to_numpy(dtype=float),
            df[self.exposure_col].to_numpy(dtype=float),
        )

    def _build_filing_report(self) -> RatePlanReport:
        report = RatePlanReport(title="Rate filing")
        if self.indication_ is not None:
            report.indication = {
                "rate_change": self.indication_.indicated_rate_change,
                "method": self.indication_.method,
                "credibility": self.indication_.credibility,
                "table": self.indication_.summary().to_html(),
            }
        if self.classification is not None:
            report.classification_relativities = {
                v: rel.to_frame() for v, rel in self.classification.relativities.items()
            }
        if self.implementation is not None:
            report.implementation_dispersion = self.implementation.dispersion_summary()
        return report

    def _excel_sheets(self) -> dict[str, pd.DataFrame]:
        sheets: dict[str, pd.DataFrame] = {}
        if self.indication_ is not None:
            sheets["indication"] = self.indication_.summary().reset_index()
        if self.classification is not None:
            sheets["relativities"] = self.classification.relativities_frame()
        if self.implementation is not None:
            sheets["impact"] = self.implementation.impacted.head(10_000).copy()
            sheets["dispersion"] = self.implementation.dispersion_summary().reset_index()
        return sheets

    def __repr__(self) -> str:
        return (
            f"RatePlan(n_policies={len(self.policies)}, "
            f"n_claims={len(self.claims)}, "
            f"indication={'yes' if self.indication_ is not None else 'no'}, "
            f"classification={'yes' if self.classification is not None else 'no'}, "
            f"implementation={'yes' if self.implementation is not None else 'no'})"
        )

    @property
    def version(self) -> str:
        return __version__
