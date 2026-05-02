import numpy as np
import pandas as pd
import pytest

from pyratemaking.trending import frequency_trend, pure_premium_trend, severity_trend


def _periods(n=8):
    return pd.Index(np.arange(2015, 2015 + n), name="ay")


def test_severity_trend_recovers_known_growth():
    idx = _periods()
    severity = 1000 * 1.04 ** (idx.astype(float) - 2015)
    counts = pd.Series([100, 110, 120, 110, 105, 115, 120, 130], index=idx)
    losses = pd.Series(severity * counts.to_numpy(), index=idx)
    fit = severity_trend(losses, counts)
    assert fit.annual_change == pytest.approx(0.04, rel=1e-9)


def test_frequency_trend_falls_when_exposure_grows_faster_than_counts():
    idx = _periods()
    exposure = pd.Series(1000 * 1.10 ** (idx.astype(float) - 2015), index=idx)
    counts = pd.Series(50 * 1.05 ** (idx.astype(float) - 2015), index=idx)
    fit = frequency_trend(counts, exposure)
    # frequency ≈ 0.05 * (1.05/1.10)^t — annual change ≈ 0.05/1.10 - 1 ≈ -4.55%
    assert fit.annual_change < 0
    assert fit.annual_change == pytest.approx(1.05 / 1.10 - 1, rel=1e-9)


def test_pure_premium_trend_consistent_with_severity_times_frequency():
    idx = _periods()
    exposure = pd.Series(1000.0, index=idx)
    counts = pd.Series(50 * 1.02 ** (idx.astype(float) - 2015), index=idx)
    severity = 1000 * 1.04 ** (idx.astype(float) - 2015)
    losses = pd.Series(severity * counts.to_numpy(), index=idx)

    sev = severity_trend(losses, counts)
    freq = frequency_trend(counts, exposure)
    pp = pure_premium_trend(losses, exposure)
    np.testing.assert_allclose(
        pp.annual_change,
        (1 + sev.annual_change) * (1 + freq.annual_change) - 1,
        rtol=1e-8,
    )


def test_severity_skips_periods_without_claims():
    idx = _periods(5)
    counts = pd.Series([100, 0, 110, 105, 120], index=idx)
    losses = pd.Series([100_000, 0, 120_000, 130_000, 150_000], index=idx).astype(float)
    fit = severity_trend(losses, counts)
    assert fit.n == 4  # the zero-count year is dropped
