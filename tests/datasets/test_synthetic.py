import numpy as np
import pytest

from pyratemaking.datasets import french_motor, synthetic


def test_synthetic_generate_deterministic_under_seed():
    a_pol, a_clm = synthetic.generate(n_policies=1000, seed=0)
    b_pol, b_clm = synthetic.generate(n_policies=1000, seed=0)
    np.testing.assert_array_equal(a_pol.to_numpy(), b_pol.to_numpy())
    np.testing.assert_array_equal(a_clm.to_numpy(), b_clm.to_numpy())


def test_synthetic_schema_matches_french_motor():
    pol, clm = synthetic.generate(n_policies=300, seed=1)
    expected = french_motor.schema()
    assert set(expected["policies"]).issubset(pol.columns)
    assert set(expected["claims"]).issubset(clm.columns)


def test_synthetic_exposure_in_range():
    pol, _ = synthetic.generate(n_policies=500, seed=1)
    assert (pol["exposure"] >= 0.05).all()
    assert (pol["exposure"] <= 1.0).all()


def test_synthetic_claim_amounts_non_negative():
    _, clm = synthetic.generate(n_policies=500, seed=2)
    if not clm.empty:
        assert (clm["claim_amount"] >= 0).all()


def test_synthetic_n_claims_matches_count_column():
    pol, clm = synthetic.generate(n_policies=400, seed=3)
    assert len(clm) == int(pol["claim_count"].sum())


def test_french_motor_schema_documented():
    schema = french_motor.schema()
    assert "policies" in schema
    assert "claims" in schema
    assert "exposure" in schema["policies"]
    assert "claim_amount" in schema["claims"]
