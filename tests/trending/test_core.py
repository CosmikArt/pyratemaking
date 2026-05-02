import numpy as np
import pandas as pd
import pytest

from pyratemaking.trending import Trend, fit_trend
from pyratemaking.trending.core import sensitivity_table


def test_multiplicative_trend_recovers_known_growth_rate():
    times = np.arange(2015, 2025)
    truth = 100 * (1.05) ** (times - 2015)
    fit = fit_trend(truth, times, kind="multiplicative")
    assert fit.annual_change == pytest.approx(0.05, rel=1e-10)
    assert fit.predict(2030) == pytest.approx(100 * 1.05**15, rel=1e-10)


def test_additive_trend_recovers_known_slope():
    times = np.arange(2015, 2025).astype(float)
    truth = 1000 + 50 * (times - 2015)
    fit = fit_trend(truth, times, kind="additive")
    assert fit.annual_change == pytest.approx(50.0, rel=1e-10)
    assert fit.predict(2025) == pytest.approx(1500.0, rel=1e-10)


def test_factor_to_brings_forward_correctly():
    times = np.arange(2018, 2025).astype(float)
    truth = 200 * 1.04 ** (times - 2018)
    fit = fit_trend(truth, times, kind="multiplicative")
    np.testing.assert_allclose(fit.factor_to(2018, 2024), 1.04**6, rtol=1e-10)


def test_confidence_interval_widens_far_from_data_centre():
    rng = np.random.default_rng(0)
    times = np.arange(2010, 2024).astype(float)
    truth = 100 * 1.03 ** (times - 2010)
    noisy = truth * np.exp(rng.normal(0, 0.05, size=times.size))
    fit = fit_trend(noisy, times, kind="multiplicative")
    near_lo, near_hi = fit.confidence_interval(2017)
    far_lo, far_hi = fit.confidence_interval(2030)
    assert (far_hi - far_lo) > (near_hi - near_lo)


def test_project_returns_predictions_with_intervals():
    times = np.arange(2015, 2024).astype(float)
    fit = fit_trend(100 * 1.04 ** (times - 2015), times, kind="multiplicative")
    df = fit.project([2025, 2026])
    assert {"predicted", "ci_lo", "ci_hi"} <= set(df.columns)
    assert (df["ci_lo"] <= df["predicted"]).all()
    assert (df["predicted"] <= df["ci_hi"]).all()


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind must be one of"):
        fit_trend([1.0, 2.0], [2020, 2021], kind="quadratic")


def test_multiplicative_rejects_non_positive_values():
    with pytest.raises(ValueError, match="strictly positive"):
        fit_trend([1.0, -1.0], [2020, 2021], kind="multiplicative")


def test_constant_times_raises():
    with pytest.raises(ValueError, match="undefined"):
        fit_trend([1.0, 2.0], [2020.0, 2020.0], kind="multiplicative")


def test_sensitivity_table_compares_forms():
    times = np.arange(2018, 2024).astype(float)
    values = 100 * 1.03 ** (times - 2018)
    out = sensitivity_table(values, times, horizon=2026)
    assert "multiplicative" in out.index
    assert "additive" in out.index
    # Multiplicative should match the underlying model better.
    expected_mult = 100 * 1.03**8
    assert out.loc["multiplicative", "projected"] == pytest.approx(expected_mult, rel=1e-9)


def test_repr_contains_kind_and_change():
    times = np.arange(2018, 2024).astype(float)
    fit = fit_trend(100 * 1.04 ** (times - 2018), times, kind="multiplicative")
    text = repr(fit)
    assert "multiplicative" in text
    assert "annual_change" in text


def test_trend_is_a_simple_dataclass():
    times = np.arange(2018, 2024).astype(float)
    fit = fit_trend(100 * 1.04 ** (times - 2018), times)
    assert isinstance(fit, Trend)
    assert fit.n == 6
