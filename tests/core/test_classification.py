import numpy as np
import pandas as pd
import pytest

from pyratemaking.core.classification import classify


def _synthetic_policies(seed: int = 0, n: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    age_band = rng.choice(["young", "mid", "senior"], size=n, p=[0.3, 0.4, 0.3])
    exposure = rng.uniform(0.4, 1.0, size=n)
    region_eff = pd.Series({"A": 0.0, "B": 0.20, "C": -0.10}).loc[region].to_numpy()
    age_eff = pd.Series({"young": 0.30, "mid": 0.0, "senior": -0.10}).loc[age_band].to_numpy()
    log_pp = np.log(800.0) + region_eff + age_eff
    pp = rng.gamma(0.5, np.exp(log_pp) / 0.5) * (rng.random(n) > 0.6)
    losses = pp * exposure
    return pd.DataFrame(
        {
            "region": region,
            "age_band": age_band,
            "exposure": exposure,
            "incurred_losses": losses,
            "claim_count": (losses > 0).astype(int),
        }
    )


def test_classify_tweedie_returns_relativities_for_each_var():
    df = _synthetic_policies()
    out = classify(
        df,
        rating_vars=["region", "age_band"],
        family="tweedie",
        backend="glum",
        base_levels={"region": "A", "age_band": "mid"},
    )
    assert "region" in out.relativities
    assert "age_band" in out.relativities
    assert out.relativities["region"].loc["A"] == pytest.approx(1.0)
    assert out.relativities["age_band"].loc["mid"] == pytest.approx(1.0)


def test_classify_balance_principle_off_balance_close_to_one_when_calibrated():
    df = _synthetic_policies()
    out = classify(
        df,
        rating_vars=["region", "age_band"],
        family="tweedie",
    )
    # default calibration: base rate makes total rated premium == total losses
    assert out.off_balance == pytest.approx(1.0, rel=1e-6)


def test_classify_predict_premium_is_positive():
    df = _synthetic_policies()
    out = classify(df, rating_vars=["region", "age_band"], family="tweedie")
    pred = out.predict_premium(df)
    assert (pred > 0).all()


def test_classify_target_average_premium_calibrates_correctly():
    df = _synthetic_policies()
    out = classify(
        df,
        rating_vars=["region", "age_band"],
        family="tweedie",
        target_average_premium=500.0,
    )
    pred = out.predict_premium(df)
    avg = pred.sum() / df["exposure"].sum()
    assert avg == pytest.approx(500.0, rel=1e-6)


def test_classify_unknown_family_raises():
    df = _synthetic_policies(n=200)
    with pytest.raises(ValueError, match="family must be"):
        classify(df, rating_vars=["region"], family="lognormal")


def test_classify_frequency_severity_requires_count_col():
    df = _synthetic_policies(n=300)
    with pytest.raises(KeyError, match="count_col"):
        classify(
            df,
            rating_vars=["region"],
            family="frequency_severity",
        )
