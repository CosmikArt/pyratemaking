"""Actual vs expected ratios at any grouping level."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def actual_vs_expected(
    df: pd.DataFrame,
    *,
    actual_col: str,
    expected_col: str,
    by: str | Sequence[str] | None = None,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Group-wise A/E ratios.

    Parameters
    ----------
    df : DataFrame
    actual_col, expected_col : str
        Columns with observed and predicted values per row.
    by : str or list of str, optional
        Grouping variables. None for overall A/E.
    weight_col : str, optional
        Per-row weight (e.g. exposure). When provided, A and E are
        weighted sums.
    """
    if by is None:
        weight = df[weight_col] if weight_col else 1.0
        actual = (df[actual_col] * weight).sum() if weight_col else df[actual_col].sum()
        expected = (df[expected_col] * weight).sum() if weight_col else df[expected_col].sum()
        return pd.DataFrame(
            {"actual": [float(actual)], "expected": [float(expected)], "a_to_e": [actual / expected]}
        )

    if isinstance(by, str):
        by = [by]
    if weight_col:
        weight = df[weight_col]
        grouped = df.assign(_a=df[actual_col] * weight, _e=df[expected_col] * weight).groupby(
            list(by), dropna=False
        )[["_a", "_e"]].sum().rename(columns={"_a": "actual", "_e": "expected"})
    else:
        grouped = df.groupby(list(by), dropna=False).agg(
            actual=(actual_col, "sum"),
            expected=(expected_col, "sum"),
        )
    grouped["a_to_e"] = grouped["actual"] / grouped["expected"].where(grouped["expected"] > 0)
    return grouped
