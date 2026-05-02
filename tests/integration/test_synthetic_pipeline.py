"""End-to-end pipeline test on the synthetic auto book."""

from pathlib import Path

import pytest

from pyratemaking import RatePlan
from pyratemaking.datasets import synthetic


@pytest.fixture(scope="module")
def book():
    pol, clm = synthetic.generate(n_policies=4_000, seed=2024)
    return pol, clm


def test_full_pipeline_produces_consistent_results(book, tmp_path: Path):
    pol, clm = book
    plan = RatePlan(pol, clm)

    indication = plan.indicate()
    assert -1 < indication.indicated_rate_change < 1

    classification = plan.classify(
        rating_vars=["region", "veh_brand", "veh_gas"],
        family="tweedie",
        backend="glum",
        power=1.5,
    )
    # Each rating variable should appear in the relativity output.
    assert set(classification.relativities.keys()) == {"region", "veh_brand", "veh_gas"}
    # Default calibration: off-balance ≈ 1.
    assert classification.off_balance == pytest.approx(1.0, rel=1e-6)

    impl = plan.implement(cap=1.20, floor=0.80)
    pct_change = impl.impacted["pct_change"].to_numpy()
    assert (pct_change <= 0.20 + 1e-9).all()
    assert (pct_change >= -0.20 - 1e-9).all()

    table = plan.diagnostics.lift(n_bins=5).table
    assert len(table) <= 5
    assert plan.diagnostics.gini() >= 0  # synthetic has signal

    out_html = plan.report.filing(tmp_path / "filing.html")
    assert out_html.exists()
    assert out_html.read_text().startswith("<!DOCTYPE html>")
    out_xlsx = plan.report.excel(tmp_path / "filing.xlsx")
    assert out_xlsx.exists()


def test_pipeline_balance_principle_holds(book):
    pol, clm = book
    plan = RatePlan(pol, clm)
    plan.classify(rating_vars=["region", "veh_brand"], family="tweedie")
    pred = plan.classification.predict_premium(pol, exposure_col="exposure")
    actual = plan._policies_with_losses["incurred_losses"].sum()
    assert pred.sum() == pytest.approx(actual, rel=1e-6)


def test_pipeline_gini_better_than_random(book):
    pol, clm = book
    plan = RatePlan(pol, clm)
    plan.classify(rating_vars=["region", "veh_brand", "driver_age"], family="tweedie")
    g = plan.diagnostics.gini(normalized=True)
    # not a tight bound — synthetic data has noise — but better than 0 is sane
    assert g > 0
