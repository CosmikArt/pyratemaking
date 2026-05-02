import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyratemaking.large_loss import (
    basic_limits_losses,
    cap_at_limit,
    increased_limits_factor_table,
    layer_loss_cost,
    layer_pricing_from_distribution,
)
from pyratemaking.large_loss.layer_pricing import layer_pricing_table


def test_cap_at_limit_basic():
    out = cap_at_limit([100.0, 500.0, 1500.0], basic_limit=1000.0)
    np.testing.assert_allclose(out, [100.0, 500.0, 1000.0])


def test_cap_at_limit_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        cap_at_limit([100.0], basic_limit=0)


def test_basic_limits_losses_attaches_excess_column():
    claims = pd.DataFrame(
        {
            "policy_ay": [2020, 2020, 2021, 2022],
            "claim_amount": [800.0, 1500.0, 200.0, 5000.0],
        }
    )
    out = basic_limits_losses(claims, basic_limit=1000.0)
    np.testing.assert_allclose(out["basic_limits_loss"].to_numpy(), [800, 1000, 200, 1000])
    np.testing.assert_allclose(out["excess_amount"].to_numpy(), [0, 500, 0, 4000])
    summary = out.attrs["summary"]
    assert summary.loc[2022, "n_capped"] == 1


def test_layer_loss_cost_unlimited_layer():
    losses = np.array([0.0, 100.0, 250.0, 600.0])
    cost = layer_loss_cost(losses, attachment=200.0, limit=None)
    # Layer payouts: 0, 0, 50, 400 → mean = 112.5
    assert cost == pytest.approx(112.5)


def test_layer_loss_cost_limited_layer():
    losses = np.array([0.0, 100.0, 250.0, 600.0])
    cost = layer_loss_cost(losses, attachment=100.0, limit=200.0)
    # Layer payouts: 0, 0, 150, 200 → mean = 87.5
    assert cost == pytest.approx(87.5)


def test_increased_limits_factors_monotone():
    rng = np.random.default_rng(0)
    losses = rng.lognormal(mean=6, sigma=1.0, size=10_000)
    table = increased_limits_factor_table(
        losses, basic_limit=1000.0, limits=[2000.0, 5000.0, 10_000.0, 25_000.0]
    )
    diffs = np.diff(table["ilf"].to_numpy())
    assert (diffs > 0).all()
    assert table.loc[1000.0, "ilf"] == pytest.approx(1.0)


def test_increased_limits_negative_loss_raises():
    with pytest.raises(ValueError, match="negative"):
        increased_limits_factor_table([-1.0, 100.0], basic_limit=50.0, limits=[100.0])


def test_layer_pricing_from_distribution_with_scipy_lognormal():
    dist = stats.lognorm(s=1.0, scale=np.exp(6))
    out = layer_pricing_from_distribution(
        distribution=dist, attachment=1000.0, limit=2000.0, n_samples=20_000
    )
    assert out["layer_loss_cost"] > 0
    assert 0 <= out["frequency"] <= 1


def test_layer_pricing_from_distribution_invalid_object_raises():
    class Bad:
        pass

    with pytest.raises(TypeError, match=".rvs"):
        layer_pricing_from_distribution(distribution=Bad(), attachment=0, limit=100)


def test_layer_pricing_table_runs():
    losses = np.array([100.0, 500.0, 1500.0, 3000.0, 10_000.0])
    table = layer_pricing_table(losses, layers=[(0, 1000.0), (1000.0, 4000.0), (5000.0, None)])
    assert len(table) == 3
    assert (table["avg_layer_loss"] >= 0).all()
