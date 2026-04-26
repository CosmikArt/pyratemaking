"""Core ratemaking components.

This module exposes the primary building blocks for a classification
ratemaking analysis following Werner & Modlin (CAS, 5th ed.).
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd


class FrequencySeverityModel:
    """GLM-based frequency–severity ratemaking model.

    Fits separate GLMs for claim frequency and claim severity, then
    combines them into a pure premium estimate.  Supports Poisson and
    Negative-Binomial frequency, Gamma and Inverse-Gaussian severity,
    and Tweedie as a single-model alternative.

    Parameters
    ----------
    freq_dist : str
        Distribution family for the frequency model.
        One of ``"poisson"``, ``"negative_binomial"``.
    sev_dist : str
        Distribution family for the severity model.
        One of ``"gamma"``, ``"inverse_gaussian"``.
    link : str, default ``"log"``
        Link function applied to both sub-models.

    References
    ----------
    Werner & Modlin, *Basic Ratemaking*, Ch. 6–9.
    Goldburd, Khare & Tevet, *GLMs for Insurance Rating*, CAS Monograph No. 5.
    """

    def __init__(
        self,
        freq_dist: str = "poisson",
        sev_dist: str = "gamma",
        link: str = "log",
    ) -> None:
        self.freq_dist = freq_dist
        self.sev_dist = sev_dist
        self.link = link
        self._freq_model: Any = None
        self._sev_model: Any = None

    def fit(
        self,
        data: pd.DataFrame,
        exposure_col: str,
        freq_target: str,
        sev_target: str,
        features: Sequence[str],
    ) -> "FrequencySeverityModel":
        """Fit frequency and severity GLMs on policy-level experience.

        Parameters
        ----------
        data : DataFrame
            Policy-level loss experience with at least the columns
            specified in *exposure_col*, *freq_target*, *sev_target*,
            and all *features*.
        exposure_col : str
            Column containing earned exposure (e.g. earned car-years).
        freq_target : str
            Column containing observed claim counts.
        sev_target : str
            Column containing incurred losses (used to derive average
            severity = incurred / claim_count).
        features : list[str]
            Rating variables to include in both sub-models.

        Returns
        -------
        self
        """
        raise NotImplementedError(
            "FrequencySeverityModel.fit() is planned for v0.1.0. "
            "See Werner & Modlin Ch. 6 for the target methodology."
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Return predicted pure premium for each record.

        Pure premium = E[frequency] × E[severity], each evaluated at
        the covariate values in *data*.
        """
        raise NotImplementedError(
            "FrequencySeverityModel.predict() is planned for v0.1.0."
        )

    def relativities(
        self, base_level: dict[str, str] | None = None
    ) -> pd.DataFrame:
        """Extract classification relativities from fitted GLM coefficients.

        Parameters
        ----------
        base_level : dict, optional
            Mapping of feature name → base level.  Relativities are
            expressed relative to this level (= 1.000).

        Returns
        -------
        DataFrame with columns ``feature``, ``level``, ``relativity``,
        ``std_error``.

        References
        ----------
        Werner & Modlin, Ch. 9 — Classification Ratemaking.
        """
        raise NotImplementedError(
            "FrequencySeverityModel.relativities() is planned for v0.1.0."
        )

    def lift_chart(self, n_bins: int = 10) -> None:
        """Plot a lift chart (observed vs predicted by model-score decile).

        Parameters
        ----------
        n_bins : int
            Number of equal-sized bins.
        """
        raise NotImplementedError(
            "Lift chart plotting is planned for v0.1.0."
        )

    def actual_vs_expected(self, by: str | Sequence[str] | None = None) -> pd.DataFrame:
        """Compute actual-vs-expected ratios, optionally grouped.

        Parameters
        ----------
        by : str or list[str], optional
            Column(s) to group by before computing A/E.

        Returns
        -------
        DataFrame with ``actual``, ``expected``, ``ae_ratio`` columns.
        """
        raise NotImplementedError(
            "A/E diagnostics are planned for v0.1.0."
        )


class BaseRateCalculator:
    """Calculate indicated base rate and overall rate level indication.

    Implements the loss-ratio and pure-premium methods described in
    Werner & Modlin Ch. 3–5, including expense loading, profit &
    contingency provisions, and credibility weighting of experience
    vs. competitor rates.

    References
    ----------
    Werner & Modlin, *Basic Ratemaking*, Ch. 3–5.
    """

    def __init__(self, method: str = "loss_ratio") -> None:
        self.method = method

    def indicated_rate_change(
        self,
        experience: pd.DataFrame,
        expense_ratio: float,
        profit_load: float = 0.0,
    ) -> float:
        """Compute the indicated overall rate change.

        Parameters
        ----------
        experience : DataFrame
            Must contain ``earned_premium``, ``incurred_loss``,
            ``loss_adjustment_expense`` columns at the desired
            aggregation level.
        expense_ratio : float
            Fixed + variable expense ratio (as a decimal).
        profit_load : float
            Target underwriting profit provision.

        Returns
        -------
        float — Indicated rate change as a decimal (e.g. 0.05 = +5 %).
        """
        raise NotImplementedError(
            "BaseRateCalculator.indicated_rate_change() is planned for v0.1.0."
        )


class ClassificationRatemaker:
    """End-to-end classification ratemaking workflow.

    Wraps GLM fitting, relativity extraction, off-balance adjustment,
    and capping into a single reproducible pipeline.

    References
    ----------
    Werner & Modlin, *Basic Ratemaking*, Ch. 9.
    """

    def __init__(self) -> None:
        pass

    def run(self, data: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Execute the full classification ratemaking pipeline.

        Parameters
        ----------
        data : DataFrame
            Policy-level experience data.
        config : dict
            Pipeline configuration (rating variables, base levels,
            capping rules, off-balance method).

        Returns
        -------
        DataFrame of final classification relativities.
        """
        raise NotImplementedError(
            "ClassificationRatemaker.run() is planned for v0.2.0."
        )


class ExposureHandler:
    """Earned and written exposure calculations.

    Handles calendar-year vs. policy-year exposure aggregation,
    in-force counts, and exposure-period alignment for trending.

    References
    ----------
    Werner & Modlin, *Basic Ratemaking*, Ch. 2, 5.
    """

    def earned_exposure(
        self,
        policies: pd.DataFrame,
        as_of: str,
        method: str = "24ths",
    ) -> pd.DataFrame:
        """Compute earned exposure by evaluation date.

        Parameters
        ----------
        policies : DataFrame
            Must contain ``effective_date``, ``expiration_date``, and
            ``written_exposure`` columns.
        as_of : str
            Evaluation date (ISO format).
        method : str
            Earning method: ``"24ths"`` (default), ``"daily"``, or
            ``"monthly"``.

        Returns
        -------
        DataFrame with ``earned_exposure`` appended.
        """
        raise NotImplementedError(
            "ExposureHandler.earned_exposure() is planned for v0.1.0."
        )


class LiftChart:
    """Standalone lift-chart builder for model comparison.

    Produces observed-vs-predicted lift charts, double lift charts
    (comparing two models), and cumulative gains curves.
    """

    def plot(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        n_bins: int = 10,
        title: str | None = None,
    ) -> None:
        """Render a lift chart comparing observed to predicted values."""
        raise NotImplementedError("LiftChart.plot() is planned for v0.1.0.")


class AEDiagnostic:
    """Actual-vs-Expected diagnostic toolkit.

    Computes A/E ratios at any grouping level and produces heat maps,
    bar charts, and summary tables used in rate filing exhibits.
    """

    def compute(
        self,
        data: pd.DataFrame,
        actual_col: str,
        expected_col: str,
        by: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Calculate A/E ratios.

        Parameters
        ----------
        data : DataFrame
        actual_col : str
            Column with observed values (losses or counts).
        expected_col : str
            Column with model-predicted values.
        by : str or list[str], optional
            Grouping variables.

        Returns
        -------
        DataFrame with ``actual``, ``expected``, ``ae_ratio`` per group.
        """
        raise NotImplementedError("AEDiagnostic.compute() is planned for v0.1.0.")
