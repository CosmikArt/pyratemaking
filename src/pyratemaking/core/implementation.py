"""Rate change implementation: caps, floors, transition rules, dislocation.

Once an indicated rate change is decided, applying it uniformly to every
policy is rare. Caps and floors limit per-policy change; transition rules
phase the change in over multiple terms; dislocation reports show how the
new rates fall across the existing book.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ImplementationResult:
    """Output of rate-change implementation."""

    impacted: pd.DataFrame
    cap: float | None
    floor: float | None

    def dispersion_summary(self, thresholds: list[float] | None = None) -> pd.DataFrame:
        """Distribution of percent rate change across policies.

        Returns the share of policies in each bin defined by ``thresholds``
        (default: ±5%, ±10%, ±15%, ±25%).
        """
        thresholds = thresholds or [-0.25, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.25]
        edges = [-np.inf, *thresholds, np.inf]
        labels = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            lo_str = f"{lo:+.0%}" if np.isfinite(lo) else "-inf"
            hi_str = f"{hi:+.0%}" if np.isfinite(hi) else "+inf"
            labels.append(f"({lo_str}, {hi_str}]")
        binned = pd.cut(self.impacted["pct_change"], bins=edges, labels=labels)
        counts = binned.value_counts(sort=False)
        return pd.DataFrame(
            {
                "n_policies": counts,
                "pct_of_book": counts / counts.sum(),
            }
        )

    def segment_summary(self, segment_col: str) -> pd.DataFrame:
        """Average and median rate change by segment."""
        if segment_col not in self.impacted.columns:
            raise KeyError(f"{segment_col!r} not in impact frame")
        return (
            self.impacted.groupby(segment_col)["pct_change"]
            .agg(["mean", "median", "min", "max", "count"])
            .rename(columns={"mean": "avg_change", "median": "median_change"})
        )

    def share_above_threshold(self, threshold: float) -> float:
        """Fraction of policies with ``pct_change > threshold``."""
        return float((self.impacted["pct_change"] > threshold).mean())

    def share_below_threshold(self, threshold: float) -> float:
        return float((self.impacted["pct_change"] < threshold).mean())

    def __repr__(self) -> str:
        return (
            f"ImplementationResult(n={len(self.impacted)}, "
            f"cap={self.cap!r}, floor={self.floor!r})"
        )


def apply_caps_floors(
    current_premium: pd.Series | np.ndarray,
    indicated_premium: pd.Series | np.ndarray,
    *,
    cap: float | None = None,
    floor: float | None = None,
) -> pd.Series:
    """Cap and floor the per-policy ratio of indicated to current premium.

    Returns the *capped* indicated premium. ``cap=1.15`` allows at most a
    +15% change per policy; ``floor=0.85`` allows at most a –15% drop.
    """
    current = np.asarray(current_premium, dtype=float)
    indicated = np.asarray(indicated_premium, dtype=float)
    if (current <= 0).any():
        raise ValueError("current premium must be strictly positive")
    if cap is not None and cap < 1:
        raise ValueError("cap must be >= 1.0")
    if floor is not None and floor > 1:
        raise ValueError("floor must be <= 1.0")

    raw_ratio = indicated / current
    if cap is not None:
        raw_ratio = np.minimum(raw_ratio, cap)
    if floor is not None:
        raw_ratio = np.maximum(raw_ratio, floor)
    return pd.Series(raw_ratio * current, name="capped_premium")


def implement_rate_change(
    policies: pd.DataFrame,
    *,
    current_premium_col: str,
    indicated_premium_col: str,
    cap: float | None = None,
    floor: float | None = None,
    extra_columns: list[str] | None = None,
) -> ImplementationResult:
    """Apply caps/floors and produce a per-policy impact frame.

    Parameters
    ----------
    policies : DataFrame
    current_premium_col, indicated_premium_col : str
        Columns with current and indicated premium.
    cap : float, optional
        Max per-policy ratio (e.g. 1.15 = +15%).
    floor : float, optional
        Min per-policy ratio (e.g. 0.85 = –15%).
    extra_columns : list of str, optional
        Additional columns to carry through onto the impact frame for later
        segment-level summaries.
    """
    current = policies[current_premium_col].to_numpy(dtype=float)
    indicated = policies[indicated_premium_col].to_numpy(dtype=float)
    capped = apply_caps_floors(current, indicated, cap=cap, floor=floor)

    cols = {
        "current_premium": current,
        "indicated_premium": indicated,
        "capped_premium": capped.to_numpy(),
    }
    if extra_columns:
        for c in extra_columns:
            if c not in policies.columns:
                raise KeyError(f"extra column {c!r} not in policies")
            cols[c] = policies[c].to_numpy()
    impact = pd.DataFrame(cols, index=policies.index)
    impact["pct_change"] = impact["capped_premium"] / impact["current_premium"] - 1.0
    impact["was_capped"] = (cap is not None) & (
        np.isclose(impact["pct_change"].to_numpy(), cap - 1) if cap is not None else False
    )
    impact["was_floored"] = (floor is not None) & (
        np.isclose(impact["pct_change"].to_numpy(), floor - 1) if floor is not None else False
    )
    return ImplementationResult(impacted=impact, cap=cap, floor=floor)
