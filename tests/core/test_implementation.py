import numpy as np
import pandas as pd
import pytest

from pyratemaking.core.implementation import (
    apply_caps_floors,
    implement_rate_change,
)


def _frame():
    return pd.DataFrame(
        {
            "policy_id": [1, 2, 3, 4, 5],
            "region": ["A", "A", "B", "B", "C"],
            "current": [1000.0, 1100.0, 900.0, 1200.0, 800.0],
            "indicated": [1300.0, 1100.0, 700.0, 1500.0, 600.0],
        }
    )


def test_apply_caps_floors_caps_at_threshold():
    df = _frame()
    capped = apply_caps_floors(df["current"], df["indicated"], cap=1.15, floor=0.85)
    assert capped.iloc[0] == pytest.approx(1000 * 1.15)  # +30% capped to +15%
    assert capped.iloc[2] == pytest.approx(900 * 0.85)  # -22% floored to -15%


def test_apply_caps_floors_validates_arguments():
    df = _frame()
    with pytest.raises(ValueError, match="cap must be"):
        apply_caps_floors(df["current"], df["indicated"], cap=0.95)
    with pytest.raises(ValueError, match="floor must be"):
        apply_caps_floors(df["current"], df["indicated"], floor=1.10)
    bad = df.copy()
    bad.loc[0, "current"] = 0
    with pytest.raises(ValueError, match="positive"):
        apply_caps_floors(bad["current"], bad["indicated"])


def test_implement_rate_change_dispersion_summary_buckets():
    df = _frame()
    result = implement_rate_change(
        df,
        current_premium_col="current",
        indicated_premium_col="indicated",
        cap=1.15,
        floor=0.85,
        extra_columns=["region"],
    )
    summary = result.dispersion_summary()
    assert summary["n_policies"].sum() == len(df)
    # share of book should sum to 1
    assert summary["pct_of_book"].sum() == pytest.approx(1.0)


def test_implement_rate_change_segment_summary():
    df = _frame()
    result = implement_rate_change(
        df,
        current_premium_col="current",
        indicated_premium_col="indicated",
        cap=1.15,
        floor=0.85,
        extra_columns=["region"],
    )
    by_region = result.segment_summary("region")
    assert set(by_region.index) == {"A", "B", "C"}
    assert "avg_change" in by_region.columns


def test_share_above_below_threshold():
    df = _frame()
    result = implement_rate_change(
        df,
        current_premium_col="current",
        indicated_premium_col="indicated",
        cap=1.15,
        floor=0.85,
    )
    assert 0.0 <= result.share_above_threshold(0.05) <= 1.0
    assert 0.0 <= result.share_below_threshold(-0.05) <= 1.0


def test_implement_rate_change_marks_capped_rows():
    df = _frame()
    result = implement_rate_change(
        df,
        current_premium_col="current",
        indicated_premium_col="indicated",
        cap=1.15,
    )
    # Policy 1 hits the cap (was +30%, now +15%).
    assert bool(result.impacted.iloc[0]["was_capped"])
