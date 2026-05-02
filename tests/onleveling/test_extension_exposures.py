import numpy as np
import pandas as pd
import pytest

from pyratemaking.onleveling import extension_of_exposures, rate_under_algorithm


def _policies():
    return pd.DataFrame(
        {
            "policy_ay": [2020, 2020, 2021, 2021, 2022],
            "earned_premium": [1000.0, 1100.0, 1050.0, 1200.0, 1400.0],
            "territory": ["A", "B", "A", "B", "A"],
        }
    )


def _algorithm(df: pd.DataFrame) -> np.ndarray:
    base = 1200.0
    relativity = df["territory"].map({"A": 1.0, "B": 1.10}).to_numpy()
    return base * relativity


def test_extension_of_exposures_recovers_current_premium_per_policy():
    pol = _policies()
    out = extension_of_exposures(pol, _algorithm)
    # AY 2020: one A (1200) + one B (1320) = 2520
    assert out.loc[2020, "on_level_premium"] == pytest.approx(2520.0)
    # AY 2022: one A only = 1200
    assert out.loc[2022, "on_level_premium"] == pytest.approx(1200.0)


def test_extension_of_exposures_factor_is_ratio():
    pol = _policies()
    out = extension_of_exposures(pol, _algorithm)
    np.testing.assert_allclose(
        out["on_level_factor"].to_numpy(),
        (out["on_level_premium"] / out["earned_premium"]).to_numpy(),
    )


def test_rate_under_algorithm_validates_length():
    pol = _policies()

    def bad(df: pd.DataFrame) -> np.ndarray:
        return np.array([1.0])

    with pytest.raises(ValueError, match="returned 1 rows for 5"):
        rate_under_algorithm(pol, bad)


def test_rate_under_algorithm_rejects_negative_premium():
    pol = _policies()

    def negative(df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), -1.0)

    with pytest.raises(ValueError, match="negative premium"):
        rate_under_algorithm(pol, negative)


def test_extension_of_exposures_zero_collected_premium_yields_nan_factor():
    pol = _policies()
    pol["earned_premium"] = 0.0
    out = extension_of_exposures(pol, _algorithm)
    assert out["on_level_factor"].isna().all()
