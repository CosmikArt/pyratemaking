from pathlib import Path

import pandas as pd
import pytest

from pyratemaking import RatePlan
from pyratemaking.datasets import synthetic


@pytest.fixture(scope="module")
def small_book():
    pol, clm = synthetic.generate(n_policies=2_000, seed=0)
    return pol, clm


def test_rateplan_constructs_from_synthetic_data(small_book):
    pol, clm = small_book
    plan = RatePlan(pol, clm)
    assert "exposure" in plan.policies.columns
    assert plan.indication_ is None
    assert plan.classification is None


def test_rateplan_indicate_loss_ratio(small_book):
    pol, clm = small_book
    plan = RatePlan(pol, clm)
    out = plan.indicate(method="loss_ratio")
    assert out.method == "loss_ratio"
    assert out.indicated_rate_change is not None


def test_rateplan_classify_and_implement(small_book):
    pol, clm = small_book
    plan = RatePlan(pol, clm)
    plan.classify(
        rating_vars=["region", "veh_brand"],
        backend="glum",
        family="tweedie",
        power=1.5,
    )
    assert plan.classification is not None
    assert "region" in plan.classification.relativities
    impl = plan.implement(cap=1.20, floor=0.80)
    assert impl is not None
    assert "pct_change" in impl.impacted.columns


def test_rateplan_diagnostics_after_classify(small_book):
    pol, clm = small_book
    plan = RatePlan(pol, clm)
    plan.classify(rating_vars=["region"], family="tweedie")
    table = plan.diagnostics.lift(n_bins=5).table
    assert len(table) <= 5
    g = plan.diagnostics.gini()
    assert -1 <= g <= 1


def test_rateplan_filing_writes_html(small_book, tmp_path: Path):
    pol, clm = small_book
    plan = RatePlan(pol, clm)
    plan.indicate()
    plan.classify(rating_vars=["region"], family="tweedie")
    plan.implement(cap=1.15, floor=0.85)
    out = plan.report.filing(tmp_path / "filing.html")
    assert out.exists()
    text = out.read_text()
    assert "Indication" in text or "indication" in text


def test_rateplan_excel_writes_workbook(small_book, tmp_path: Path):
    pol, clm = small_book
    plan = RatePlan(pol, clm)
    plan.indicate()
    plan.classify(rating_vars=["region"], family="tweedie")
    plan.implement(cap=1.15, floor=0.85)
    out = plan.report.excel(tmp_path / "filing.xlsx")
    assert out.exists()


def test_rateplan_implement_requires_classify(small_book):
    pol, clm = small_book
    plan = RatePlan(pol, clm)
    with pytest.raises(RuntimeError, match="classify"):
        plan.implement(cap=1.15)


def test_rateplan_repr_describes_state(small_book):
    pol, clm = small_book
    plan = RatePlan(pol, clm)
    text = repr(plan)
    assert "RatePlan" in text
    assert "indication=no" in text
    assert "classification=no" in text
