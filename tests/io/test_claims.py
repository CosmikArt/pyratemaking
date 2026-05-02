import numpy as np
import pandas as pd
import pytest

from pyratemaking.io.claims import (
    ClaimsSchema,
    aggregate_to_ay,
    loss_triangle,
    merge_policy_losses,
    validate_claims,
)


def _claims():
    return pd.DataFrame(
        {
            "policy_id": [1, 1, 2, 3],
            "policy_ay": [2020, 2020, 2020, 2021],
            "claim_amount": [500.0, 0.0, 1500.0, 800.0],
        }
    )


def test_validate_claims_happy_path():
    out = validate_claims(_claims())
    assert out["claim_amount"].dtype == float
    assert out["policy_ay"].dtype == int


def test_validate_claims_negative_loss_raises():
    df = _claims()
    df.loc[0, "claim_amount"] = -10
    with pytest.raises(ValueError, match="negative"):
        validate_claims(df)


def test_validate_claims_drops_zero_loss_when_disabled():
    out = validate_claims(_claims(), allow_zero_loss=False)
    assert (out["claim_amount"] > 0).all()


def test_aggregate_to_ay_returns_sum_and_count():
    out = aggregate_to_ay(validate_claims(_claims()))
    assert out.loc[2020, "incurred_losses"] == pytest.approx(2000.0)
    assert out.loc[2020, "claim_count"] == 2  # the zero-amount row excluded
    assert out.loc[2021, "claim_count"] == 1


def test_merge_policy_losses_attaches_zero_for_unmatched_policies():
    pol = pd.DataFrame({"policy_id": [1, 2, 3, 4], "exposure": [1, 1, 1, 1]})
    out = merge_policy_losses(pol, validate_claims(_claims()))
    assert out.loc[out["policy_id"] == 4, "incurred_losses"].iloc[0] == 0.0
    assert out.loc[out["policy_id"] == 1, "incurred_losses"].iloc[0] == 500.0


def test_loss_triangle_cumulative_matches_manual():
    claims = pd.DataFrame(
        {
            "policy_ay": [2020, 2020, 2020, 2021, 2021],
            "development_age": [12, 24, 36, 12, 24],
            "claim_amount": [100.0, 50.0, 25.0, 200.0, 75.0],
        }
    )
    tri = loss_triangle(claims, cumulative=True)
    assert tri.loc[2020, 12] == pytest.approx(100.0)
    assert tri.loc[2020, 24] == pytest.approx(150.0)
    assert tri.loc[2020, 36] == pytest.approx(175.0)
    assert tri.loc[2021, 24] == pytest.approx(275.0)
    assert np.isnan(tri.loc[2021, 36])
