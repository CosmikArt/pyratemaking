import numpy as np
import pandas as pd
import pytest

from pyratemaking.diagnostics import (
    LiftChart,
    actual_vs_expected,
    decile_analysis,
    deviance_residuals,
    double_lift,
    gini_coefficient,
    lift_table,
    lorenz_curve,
    partial_dependence,
    pearson_residuals,
    reliability_diagram,
)
from pyratemaking.diagnostics.pdp import accumulated_local_effects


def _data():
    rng = np.random.default_rng(0)
    n = 5000
    pred = rng.uniform(0.05, 0.5, size=n)
    actual = rng.poisson(pred)
    weight = rng.uniform(0.5, 1.5, size=n)
    return actual, pred, weight


def test_lift_table_monotone_for_perfect_model():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 1, size=2000)
    actual = pred  # perfect alignment
    table = lift_table(actual, pred, n_bins=10)
    assert table["avg_actual"].is_monotonic_increasing


def test_lift_table_returns_correct_number_of_bins():
    actual, pred, weight = _data()
    table = lift_table(actual, pred, weights=weight, n_bins=10)
    assert len(table) == 10


def test_double_lift_runs_without_error():
    actual, pred, _ = _data()
    pred_b = pred * np.random.default_rng(1).uniform(0.8, 1.2, size=len(pred))
    out = double_lift(actual, pred, pred_b)
    assert "avg_pred_a" in out.columns
    assert "avg_pred_b" in out.columns


def test_double_lift_rejects_non_positive_pred_b():
    actual = np.array([1.0, 2.0])
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="pred_b"):
        double_lift(actual, a, b)


def test_decile_analysis_includes_a_to_e():
    actual, pred, _ = _data()
    out = decile_analysis(actual, pred)
    assert "a_to_e" in out.columns


def test_actual_vs_expected_grouped():
    df = pd.DataFrame(
        {
            "region": ["A", "A", "B", "B", "C"],
            "actual": [10, 12, 5, 6, 8],
            "expected": [11, 11, 6, 5, 7],
        }
    )
    out = actual_vs_expected(df, actual_col="actual", expected_col="expected", by="region")
    assert set(out.index) == {"A", "B", "C"}
    assert out.loc["A", "a_to_e"] == pytest.approx(22 / 22)


def test_actual_vs_expected_overall():
    df = pd.DataFrame({"a": [1, 2, 3], "e": [2, 2, 2]})
    out = actual_vs_expected(df, actual_col="a", expected_col="e")
    assert out.loc[0, "a_to_e"] == pytest.approx(6 / 6)


def test_gini_coefficient_for_perfect_predictor_close_to_max():
    rng = np.random.default_rng(0)
    actual = rng.uniform(0, 1, size=1000)
    perfect = actual
    g = gini_coefficient(actual, perfect)
    constant = gini_coefficient(actual, np.ones_like(actual))
    assert g > constant


def test_gini_normalized_in_zero_one():
    rng = np.random.default_rng(0)
    actual = rng.poisson(0.1, size=2000)
    pred = actual + rng.normal(0, 0.05, size=2000)
    g = gini_coefficient(actual, pred, normalized=True)
    assert -0.05 <= g <= 1.05


def test_lorenz_curve_endpoints():
    actual = np.array([1, 2, 3, 4])
    pred = np.array([1, 2, 3, 4])
    curve = lorenz_curve(actual, pred)
    assert curve["cum_weight"].iloc[-1] == pytest.approx(1.0)
    assert curve["cum_actual"].iloc[-1] == pytest.approx(1.0)


def test_pearson_residuals_poisson_known_value():
    y = np.array([2.0, 3.0])
    mu = np.array([1.0, 4.0])
    r = pearson_residuals(y, mu, family="poisson")
    np.testing.assert_allclose(r, (y - mu) / np.sqrt(mu))


def test_deviance_residuals_signed():
    y = np.array([0.0, 5.0])
    mu = np.array([2.0, 2.0])
    r = deviance_residuals(y, mu, family="poisson")
    assert r[0] < 0
    assert r[1] > 0


def test_partial_dependence_for_linear_predictor():
    X = pd.DataFrame({"x": np.linspace(0, 10, 100), "z": np.zeros(100)})
    pdp = partial_dependence(lambda df: df["x"].to_numpy() * 2, X, feature="x")
    # PDP should track the linear effect
    assert pdp["pdp"].is_monotonic_increasing


def test_accumulated_local_effects_monotone():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(0, 10, size=400), "z": rng.normal(size=400)})
    out = accumulated_local_effects(lambda df: df["x"].to_numpy() ** 2, X, "x", n_bins=10)
    assert out["ale"].diff().dropna().ge(-1e-6).all()


def test_reliability_diagram_returns_bias_column():
    actual, pred, _ = _data()
    out = reliability_diagram(actual, pred, n_bins=10)
    assert "bias" in out.columns


def test_lift_chart_figure_renders():
    actual, pred, _ = _data()
    table = lift_table(actual, pred)
    fig = LiftChart(table).figure(title="test")
    assert fig is not None
    assert fig.axes[0].get_title() == "test"
