import numpy as np
import pandas as pd
import pytest

from pyratemaking.onleveling import (
    RateChange,
    average_rate_level,
    on_level_factors,
    parallelogram,
)
from pyratemaking.onleveling.parallelogram import to_fractional_year


def test_no_rate_changes_yields_unity():
    avg = average_rate_level(2020, rate_changes=[], starting_factor=1.0)
    assert avg == pytest.approx(1.0)


def test_change_at_year_start_affects_lower_right_triangle_only():
    # A change at u=Y catches only the right half of the hat (area 0.5).
    avg = average_rate_level(
        2020, rate_changes=[RateChange(date=2020.0, factor=1.10)]
    )
    assert avg == pytest.approx(0.5 * 1.0 + 0.5 * 1.10, rel=1e-12)


def test_change_at_prior_year_start_affects_full_year():
    # A change at u=Y-1 catches the entire hat.
    avg = average_rate_level(
        2020, rate_changes=[RateChange(date=2019.0, factor=1.10)]
    )
    assert avg == pytest.approx(1.10, rel=1e-12)


def test_single_mid_year_change_splits_proportionally():
    # Change at 2020.5 catches only the upper-right triangle of the hat:
    # base 0.5 wide × height 0.5 / 2 = 0.125 area.
    avg = average_rate_level(
        2020, rate_changes=[RateChange(date=2020.5, factor=1.10)]
    )
    assert avg == pytest.approx(0.875 * 1.0 + 0.125 * 1.10, rel=1e-12)


def test_two_changes_textbook_geometry():
    # CY 2011 with changes at 2010.5 (+5%) and 2011.25 (+3%).
    # Worked out by hand in the module docstring.
    avg = average_rate_level(
        2011,
        rate_changes=[
            RateChange(date=2010.5, factor=1.05),
            RateChange(date=2011.25, factor=1.03),
        ],
        starting_factor=1.0,
    )
    expected = 0.125 * 1.0 + 0.59375 * 1.05 + 0.28125 * (1.05 * 1.03)
    assert avg == pytest.approx(expected, rel=1e-12)


def test_parallelogram_constant_exposure_returns_unity_when_no_changes():
    ep = pd.Series([1000.0, 1000.0, 1000.0], index=[2018, 2019, 2020])
    out = parallelogram(ep, rate_changes=[])
    np.testing.assert_allclose(out["on_level_factor"].to_numpy(), 1.0)
    np.testing.assert_allclose(out["on_level_premium"].to_numpy(), 1000.0)


def test_parallelogram_brings_history_to_current_level():
    ep = pd.Series([1_000_000.0, 1_200_000.0, 1_300_000.0], index=[2019, 2020, 2021])
    changes = [RateChange(date=2020.5, factor=1.10), RateChange(date=2021.5, factor=1.05)]
    out = parallelogram(ep, changes)
    current_factor = 1.10 * 1.05
    assert out["on_level_factor"].iloc[0] == pytest.approx(current_factor / 1.0)
    last_avg = average_rate_level(2021, changes)
    assert out["on_level_factor"].iloc[2] == pytest.approx(current_factor / last_avg)


def test_on_level_factors_helper_matches_parallelogram_output():
    changes = [RateChange(date=2020.5, factor=1.10)]
    factors = on_level_factors([2018, 2019, 2020, 2021], changes)
    ep = pd.Series([1.0, 1.0, 1.0, 1.0], index=[2018, 2019, 2020, 2021])
    via_parallelogram = parallelogram(ep, changes)["on_level_factor"]
    np.testing.assert_allclose(factors.to_numpy(), via_parallelogram.to_numpy())


def test_to_fractional_year_known_dates():
    assert to_fractional_year("2020-01-01") == pytest.approx(2020.0)
    # 2020 is a leap year — Jul 1 = day 183 of 366
    assert to_fractional_year("2020-07-01") == pytest.approx(2020 + 182 / 366, rel=1e-9)
    assert to_fractional_year("2021-01-01") == pytest.approx(2021.0)
