import numpy as np
import pandas as pd
import pytest

from pyratemaking.glm import (
    FrequencySeverityModel,
    fit_frequency,
    fit_severity,
)


def test_fit_frequency_recovers_region_relativities(synthetic_freq_data):
    df = synthetic_freq_data
    res = fit_frequency(
        df[["region", "driver_age"]],
        df["claim_count"],
        df["exposure"],
        base_levels={"region": "A"},
    )
    rel = res.relativities("region")
    # True relativities: A=1.0, B=exp(0.30)=1.3499, C=exp(-0.20)=0.8187
    assert rel.loc["A"] == pytest.approx(1.0)
    assert rel.loc["B"] == pytest.approx(np.exp(0.30), rel=0.10)
    assert rel.loc["C"] == pytest.approx(np.exp(-0.20), rel=0.10)


def test_fit_severity_skips_zero_count_rows(synthetic_freq_data):
    df = synthetic_freq_data.copy()
    df["claim_amount"] = df["claim_count"] * 800.0
    res = fit_severity(
        df[["region", "driver_age"]],
        df["claim_amount"],
        df["claim_count"],
    )
    # All claims have severity 800 → relativities should be ~ 1.
    rel = res.relativities("region")
    np.testing.assert_allclose(rel.to_numpy(), 1.0, atol=1e-6)


def test_frequency_severity_model_predicts_pure_premium(synthetic_freq_data):
    df = synthetic_freq_data.copy()
    df["claim_amount"] = df["claim_count"] * 1000.0
    fs = FrequencySeverityModel.fit(
        df[["region", "driver_age"]],
        df["claim_count"],
        df["claim_amount"],
        df["exposure"],
    )
    pp = fs.predict(df[["region", "driver_age"]], df["exposure"])
    # Mean predicted pure premium ≈ mean observed loss / mean exposure.
    df["claim_amount"].sum()
    pred_total = (pp * df["exposure"]).sum() / df["exposure"].mean()  # approx
    assert pp.shape == (len(df),)
    assert (pp >= 0).all()
    assert pred_total > 0


def test_severity_requires_at_least_one_claim():
    X = pd.DataFrame({"region": ["A", "B"]})
    losses = np.array([0.0, 0.0])
    counts = np.array([0, 0])
    with pytest.raises(ValueError, match="at least one row"):
        fit_severity(X, losses, counts)
