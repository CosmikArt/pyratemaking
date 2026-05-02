"""End-to-end tests of optional bridges to the actuarial ecosystem.

These tests are skipped when the optional package is not installed. To run
them locally:

    pip install -e ".[full]"
    pytest tests/integration/test_ecosystem_bridges.py -v

In CI, the ecosystem matrix job installs the [full] extra and runs these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# -----------------------------------------------------------------------------
# whsmooth bridge — relativities/smoothing.py method="whittaker"
# -----------------------------------------------------------------------------


class TestWhsmoothBridge:
    """Verify the smooth_relativities -> whsmooth bridge end-to-end."""

    @pytest.fixture
    def noisy_relativities(self):
        rng = np.random.default_rng(0)
        ages = np.arange(18, 90, 6)
        true_rel = 0.6 + 0.015 * (ages - 50) ** 2 / 100
        observed = true_rel * np.exp(rng.normal(0, 0.08, len(ages)))
        return (
            pd.Series(observed, index=ages, name="rel"),
            pd.Series(true_rel, index=ages),
        )

    def test_whittaker_method_callable(self, noisy_relativities):
        pytest.importorskip("whsmooth")
        from pyratemaking.relativities.smoothing import smooth_relativities

        observed, _ = noisy_relativities
        smoothed = smooth_relativities(observed, method="whittaker", smoothing_param=10.0)
        assert isinstance(smoothed, pd.Series)
        assert len(smoothed) == len(observed)
        assert smoothed.notna().all()
        # Index preserved.
        np.testing.assert_array_equal(smoothed.index.to_numpy(), observed.index.to_numpy())

    def test_whittaker_reduces_noise(self, noisy_relativities):
        pytest.importorskip("whsmooth")
        from pyratemaking.relativities.smoothing import smooth_relativities

        observed, _truth = noisy_relativities
        smoothed = smooth_relativities(observed, method="whittaker", smoothing_param=5.0)
        # Smoothed series should have smaller second-difference roughness than raw.
        rough_raw = float(np.sum(np.diff(observed.to_numpy(), n=2) ** 2))
        rough_smoothed = float(np.sum(np.diff(smoothed.to_numpy(), n=2) ** 2))
        assert rough_smoothed < rough_raw, (
            f"Whittaker should smooth; raw_roughness={rough_raw:.5f} "
            f"smoothed_roughness={rough_smoothed:.5f}"
        )

    def test_whittaker_higher_lambda_smoother(self, noisy_relativities):
        pytest.importorskip("whsmooth")
        from pyratemaking.relativities.smoothing import smooth_relativities

        observed, _ = noisy_relativities
        low = smooth_relativities(observed, method="whittaker", smoothing_param=1.0)
        high = smooth_relativities(observed, method="whittaker", smoothing_param=1e4)
        roughness_low = float(np.sum(np.diff(low.to_numpy(), n=2) ** 2))
        roughness_high = float(np.sum(np.diff(high.to_numpy(), n=2) ** 2))
        assert roughness_high < roughness_low

    def test_moving_average_does_not_require_whsmooth(self, noisy_relativities):
        from pyratemaking.relativities.smoothing import smooth_relativities

        observed, _ = noisy_relativities
        result = smooth_relativities(observed, method="moving_average", window=3)
        assert len(result) == len(observed)


# -----------------------------------------------------------------------------
# actuarcredibility bridge — relativities/credibility.py method="actuarcredibility"
# -----------------------------------------------------------------------------


class TestActuarCredibilityBridge:
    @pytest.fixture
    def relativities_with_exposure(self):
        idx = pd.Index(["A", "B", "C", "D", "E"], name="territory")
        rels = pd.Series([0.85, 0.95, 1.00, 1.10, 1.30], index=idx)
        expo = pd.Series([100.0, 500.0, 5000.0, 200.0, 50.0], index=idx)
        return rels, expo

    def test_actuarcredibility_method_callable(self, relativities_with_exposure):
        pytest.importorskip("actuarcredibility")
        from pyratemaking.relativities.credibility import credibility_weighted

        rels, expo = relativities_with_exposure
        out = credibility_weighted(rels, exposures=expo, method="actuarcredibility")
        assert isinstance(out, pd.Series)
        assert len(out) == len(rels)
        assert out.notna().all()

    def test_high_exposure_pulls_z_toward_1(self, relativities_with_exposure):
        pytest.importorskip("actuarcredibility")
        from pyratemaking.relativities.credibility import credibility_weighted

        rels, expo = relativities_with_exposure
        weighted = credibility_weighted(rels, exposures=expo, method="actuarcredibility")
        # Cell with 5000 exposure should retain most of its raw relativity;
        # cell with 50 should be pulled hard toward the complement (1.0).
        assert abs(weighted.loc["C"] - rels.loc["C"]) < abs(weighted.loc["E"] - rels.loc["E"])

    def test_actuarcredibility_uses_lib_classical_formula(self, relativities_with_exposure):
        """The bridge result should match the classical sqrt rule (same formula)."""
        pytest.importorskip("actuarcredibility")
        from pyratemaking.relativities.credibility import credibility_weighted

        rels, expo = relativities_with_exposure
        via_lib = credibility_weighted(rels, exposures=expo, method="actuarcredibility")
        via_builtin = credibility_weighted(rels, exposures=expo, method="square_root")
        np.testing.assert_allclose(via_lib.to_numpy(), via_builtin.to_numpy(), rtol=1e-10)

    def test_square_root_default_works_without_actuarcredibility(self, relativities_with_exposure):
        from pyratemaking.relativities.credibility import credibility_weighted

        rels, expo = relativities_with_exposure
        out = credibility_weighted(rels, exposures=expo, method="square_root")
        assert isinstance(out, pd.Series)


# -----------------------------------------------------------------------------
# burncost bridge — development/_burncost_bridge.py
# -----------------------------------------------------------------------------


class TestBurncostBridge:
    @pytest.fixture
    def small_triangle(self):
        data = np.array(
            [
                [1000, 1500, 1700, 1750],
                [1200, 1750, 1950, np.nan],
                [1100, 1600, np.nan, np.nan],
                [1300, np.nan, np.nan, np.nan],
            ],
            dtype=float,
        )
        return pd.DataFrame(
            data,
            index=[2022, 2023, 2024, 2025],
            columns=[12, 24, 36, 48],
        )

    def test_to_burncost_returns_loss_triangle(self, small_triangle):
        burncost = pytest.importorskip("burncost")
        from pyratemaking.development._burncost_bridge import to_burncost_triangle

        out = to_burncost_triangle(small_triangle)
        assert isinstance(out, burncost.LossTriangle)
        assert out.accident_years == [2022, 2023, 2024, 2025]
        assert out.dev_periods == [12, 24, 36, 48]
        assert out.values.shape == (4, 4)

    def test_chainladder_to_burncost_method(self, small_triangle):
        pytest.importorskip("burncost")
        from pyratemaking.development.ldf import ChainLadder

        cl = ChainLadder(small_triangle)
        bc_triangle = cl.to_burncost()
        assert bc_triangle is not None
        assert hasattr(bc_triangle, "values")


# -----------------------------------------------------------------------------
# Smoke test: full pipeline with [full] extras enabled
# -----------------------------------------------------------------------------


class TestEcosystemPipelineSmoke:
    """One end-to-end pipeline that exercises the bridges in sequence."""

    def test_synthetic_pipeline_with_ecosystem(self):
        pytest.importorskip("whsmooth")
        pytest.importorskip("actuarcredibility")
        pytest.importorskip("burncost")

        from pyratemaking import RatePlan
        from pyratemaking.datasets import synthetic
        from pyratemaking.relativities.credibility import credibility_weighted
        from pyratemaking.relativities.smoothing import smooth_relativities

        policies, claims = synthetic.generate(n_policies=2_000, seed=42)

        plan = RatePlan(
            policies=policies,
            claims=claims,
            exposure_col="exposure",
            loss_col="claim_amount",
            ay_col="policy_ay",
        )
        indication = plan.indicate(method="loss_ratio", target_lr=0.65)
        assert indication is not None
        assert -1 < indication.indicated_rate_change < 1

        plan.classify(
            rating_vars=["region", "veh_brand"],
            family="tweedie",
            backend="glum",
            power=1.5,
        )

        # Exercise both bridges on the fitted relativities.
        region_rel = plan.classification.relativities["region"]
        region_expo = policies.groupby("region")["exposure"].sum()
        weighted = credibility_weighted(region_rel, region_expo, method="actuarcredibility")
        assert len(weighted) == len(region_rel)

        smoothed = smooth_relativities(region_rel, method="whittaker", smoothing_param=5.0)
        assert len(smoothed) == len(region_rel)
