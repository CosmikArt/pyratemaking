"""Pandas Styler helpers for actuarial output.

These functions return ``Styler`` objects (HTML/Jupyter-aware) with the
formatting conventions used in CAS rate filings: relativities to four
decimals, percentages with sign, currency with thousands separators.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def format_currency(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"${x:,.0f}"


def format_percent(x: float, *, decimals: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.{decimals}%}"


def format_relativity(x: float, *, decimals: int = 4) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


def style_actuarial_table(
    df: pd.DataFrame,
    *,
    currency_cols: Sequence[str] = (),
    percent_cols: Sequence[str] = (),
    relativity_cols: Sequence[str] = (),
    title: str | None = None,
):
    """Return a pandas ``Styler`` with the requested column formats applied."""
    fmt: dict[str, object] = {}
    for c in currency_cols:
        if c in df.columns:
            fmt[c] = format_currency
    for c in percent_cols:
        if c in df.columns:
            fmt[c] = format_percent
    for c in relativity_cols:
        if c in df.columns:
            fmt[c] = format_relativity

    styler = df.style.format(fmt) if fmt else df.style
    styler = styler.set_table_attributes('class="actuarial-table"')
    if title is not None:
        styler = styler.set_caption(title)
    return styler
