import numpy as np
import pandas as pd
import pytest

from pyratemaking.io.policies import (
    PolicySchema,
    aggregate_to_ay,
    attach_pure_premium,
    validate_policies,
)


def _frame():
    return pd.DataFrame(
        {
            "policy_id": [1, 2, 3, 4],
            "policy_ay": [2020, 2020, 2021, 2021],
            "exposure": [1.0, 0.5, 1.0, 0.0],
            "earned_premium": [1200.0, 600.0, 1300.0, 0.0],
        }
    )


def test_validate_policies_happy_path():
    out = validate_policies(_frame())
    assert out["exposure"].dtype == float
    assert out["policy_ay"].dtype == int


def test_validate_policies_drops_zero_exposure_when_disabled():
    out = validate_policies(_frame(), allow_zero_exposure=False)
    assert (out["exposure"] > 0).all()
    assert len(out) == 3


def test_validate_policies_negative_exposure_raises():
    df = _frame()
    df.loc[0, "exposure"] = -0.1
    with pytest.raises(ValueError, match="negative"):
        validate_policies(df)


def test_validate_policies_missing_required_column_raises():
    df = _frame().drop(columns=["exposure"])
    with pytest.raises(KeyError, match="exposure"):
        validate_policies(df)


def test_validate_policies_missing_optional_premium_is_dropped_silently():
    df = _frame().drop(columns=["earned_premium"])
    out = validate_policies(df)
    assert "earned_premium" not in out.columns


def test_aggregate_to_ay_sums_exposure_and_premium():
    out = aggregate_to_ay(validate_policies(_frame()))
    assert out.loc[2020, "exposure"] == pytest.approx(1.5)
    assert out.loc[2021, "earned_premium"] == pytest.approx(1300.0)


def test_attach_pure_premium_handles_zero_exposure():
    df = validate_policies(_frame())
    losses = np.array([300.0, 100.0, 400.0, 0.0])
    out = attach_pure_premium(df, losses)
    assert out.loc[3, "pure_premium"] == 0.0
    assert out.loc[0, "pure_premium"] == pytest.approx(300.0)


def test_custom_schema_overrides_column_names():
    df = pd.DataFrame(
        {
            "id": [1],
            "ay": [2020],
            "exp": [1.0],
            "ep": [100.0],
        }
    )
    schema = PolicySchema(exposure="exp", ay="ay", premium="ep", policy_id="id")
    out = validate_policies(df, schema)
    assert out["exp"].iloc[0] == 1.0
