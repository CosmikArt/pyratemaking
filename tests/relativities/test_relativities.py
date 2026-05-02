import numpy as np
import pandas as pd
import pytest

from pyratemaking.glm import fit_frequency
from pyratemaking.relativities import (
    credibility_weighted,
    multi_way_relativities,
    one_way_relativities,
    smooth_relativities,
)
from pyratemaking.relativities.multi_way import (
    balance_principle_check,
    relativities_to_frame,
)


def _toy_policies():
    return pd.DataFrame(
        {
            "region": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "exposure": [1.0] * 12,
            "incurred_losses": [
                100.0,
                110.0,
                90.0,
                100.0,  # A
                150.0,
                160.0,
                140.0,
                150.0,  # B
                70.0,
                80.0,
                60.0,
                90.0,  # C
            ],
            "claim_count": [1] * 12,
        }
    )


def test_one_way_relativities_uses_largest_exposure_as_default_base():
    df = _toy_policies()
    out = one_way_relativities(df, "region", count_col="claim_count")
    assert out.loc["A", "relativity"] == pytest.approx(1.0)
    assert out.loc["B", "relativity"] == pytest.approx(1.5)
    assert out.loc["C", "relativity"] == pytest.approx(0.75)


def test_one_way_relativities_explicit_base_level():
    df = _toy_policies()
    out = one_way_relativities(df, "region", base_level="B")
    assert out.loc["B", "relativity"] == pytest.approx(1.0)


def test_multi_way_relativities_pulled_from_glm():
    rng = np.random.default_rng(0)
    n = 8000
    region = rng.choice(["A", "B"], size=n)
    exposure = rng.uniform(0.5, 1.0, size=n)
    log_lambda = np.where(region == "A", np.log(0.10), np.log(0.10) + 0.20)
    counts = rng.poisson(np.exp(log_lambda) * exposure)
    df = pd.DataFrame({"region": region, "exposure": exposure, "claim_count": counts})
    fit = fit_frequency(df[["region"]], df["claim_count"], df["exposure"])
    rel = multi_way_relativities(fit, ["region"])
    assert rel["region"].loc["A"] == pytest.approx(1.0)
    assert rel["region"].loc["B"] == pytest.approx(np.exp(0.20), rel=0.10)


def test_relativities_to_frame_stacks_correctly():
    rels = {
        "region": pd.Series({"A": 1.0, "B": 1.2}),
        "vehicle_age": pd.Series({1: 1.0, 5: 0.9}),
    }
    df = relativities_to_frame(rels)
    assert set(df["variable"]) == {"region", "vehicle_age"}
    assert len(df) == 4


def test_balance_principle_check_returns_off_balance_ratio():
    df = pd.DataFrame(
        {
            "region": ["A", "A", "B", "B"],
            "exposure": [1.0, 1.0, 1.0, 1.0],
            "incurred_losses": [100, 110, 150, 160],
        }
    )
    rels = {"region": pd.Series({"A": 1.0, "B": 1.5})}
    out = balance_principle_check(df, rels, base_rate=100.0)
    expected_premium = 100 * (1.0 + 1.0 + 1.5 + 1.5)
    assert out["rated_premium"] == pytest.approx(expected_premium)


def test_credibility_weighted_blends_with_complement():
    rel = pd.Series({"A": 1.20, "B": 0.80})
    exposure = pd.Series({"A": 100.0, "B": 100.0})
    out = credibility_weighted(rel, exposure, complement=1.0, full_credibility_exposure=400.0)
    z = np.sqrt(100 / 400)
    assert out.loc["A"] == pytest.approx(z * 1.20 + (1 - z) * 1.0)
    assert out.loc["B"] == pytest.approx(z * 0.80 + (1 - z) * 1.0)


def test_smooth_relativities_moving_average_smooths_kink():
    rel = pd.Series([1.0, 1.5, 1.0, 1.5, 1.0, 1.5], index=range(6))
    smoothed = smooth_relativities(rel, window=3)
    # the centred moving average should compress the variation
    assert smoothed.std() < rel.std()


def test_smooth_relativities_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown smoothing"):
        smooth_relativities(pd.Series([1.0, 1.1]), method="kernel")
