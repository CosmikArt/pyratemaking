"""End-to-end pipeline test on the French Motor dataset.

Network-dependent — marked ``slow`` and ``network``. Skipped automatically
when the dataset is not cached locally and no network is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyratemaking import RatePlan
from pyratemaking.datasets import french_motor


def _cache_present() -> bool:
    from pyratemaking.datasets._cache import cache_dir

    cache = cache_dir()
    return (cache / "freMTPL2freq.csv").exists() and (cache / "freMTPL2sev.csv").exists()


pytestmark = [
    pytest.mark.slow,
    pytest.mark.network,
]


@pytest.fixture(scope="module")
def french_book():
    if not _cache_present():
        try:
            return french_motor.load()
        except OSError as exc:
            pytest.skip(f"french motor data unavailable: {exc}")
    return french_motor.load()


def test_french_motor_pipeline_runs(french_book, tmp_path: Path):
    policies, claims = french_book
    sample = policies.sample(20_000, random_state=0)
    sample = sample.reset_index(drop=True)
    sample_claims = claims[claims["policy_id"].isin(sample["policy_id"])].reset_index(
        drop=True
    )
    plan = RatePlan(sample, sample_claims, premium_col=None)
    # No earned premium column — use pure-premium indication.
    plan.indicate(
        method="pure_premium",
        current_average_rate=400.0,
        fixed_expense_per_exposure=20.0,
    )
    plan.classify(
        rating_vars=["region", "area", "veh_gas"],
        family="tweedie",
        backend="glum",
        power=1.5,
    )
    impl = plan.implement(cap=1.25, floor=0.75, current_premium_col="exposure")
    assert (impl.impacted["pct_change"] <= 0.25 + 1e-9).all()
    assert (impl.impacted["pct_change"] >= -0.25 - 1e-9).all()
