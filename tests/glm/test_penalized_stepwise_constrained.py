import numpy as np
import pandas as pd

from pyratemaking.glm import (
    fit_monotone_glm,
    fit_penalized,
    monotone_relativities,
    stepwise_select,
)


def test_fit_penalized_ridge_runs(synthetic_pure_premium_data):
    df = synthetic_pure_premium_data
    res = fit_penalized(
        df[["region", "driver_age"]],
        df["pure_premium"],
        family="tweedie",
        link="log",
        tweedie_power=1.5,
        penalty="ridge",
        alpha=0.01,
        sample_weight=df["exposure"],
    )
    assert res.n_obs == len(df)
    assert "Intercept" in res.coef_.index


def test_fit_penalized_cv_picks_alpha_from_grid(synthetic_pure_premium_data):
    df = synthetic_pure_premium_data
    res = fit_penalized(
        df[["region", "driver_age"]],
        df["pure_premium"],
        family="tweedie",
        link="log",
        penalty="elastic_net",
        alpha="cv",
        cv_alphas=(0.001, 0.01, 0.1),
        cv_folds=3,
        sample_weight=df["exposure"],
    )
    assert res.coef_.shape[0] >= 1


def test_stepwise_forward_picks_a_variable(synthetic_freq_data):
    df = synthetic_freq_data
    out = stepwise_select(
        df[["region", "driver_age"]],
        df["claim_count"],
        candidates=["region", "driver_age"],
        direction="forward",
        criterion="aic",
        family="poisson",
        link="log",
        exposure=df["exposure"],
    )
    assert "region" in out.selected
    assert len(out.history) >= 2


def test_monotone_relativities_projection_increasing():
    rel = pd.Series([1.0, 1.05, 0.95, 1.10], index=[20, 30, 40, 50])
    proj = monotone_relativities(rel, increasing=True)
    diffs = np.diff(proj.to_numpy())
    assert (diffs >= -1e-12).all()


def test_monotone_relativities_decreasing():
    rel = pd.Series([1.5, 1.2, 1.3, 0.9], index=[1, 2, 3, 4])
    proj = monotone_relativities(rel, increasing=False)
    diffs = np.diff(proj.to_numpy())
    assert (diffs <= 1e-12).all()


def test_fit_monotone_glm_runs_without_error(synthetic_freq_data):
    df = synthetic_freq_data.copy()
    df["age_band"] = pd.cut(
        df["driver_age"], bins=[0, 25, 35, 45, 55, 100], labels=["a", "b", "c", "d", "e"]
    ).astype(str)
    res = fit_monotone_glm(
        df[["region", "age_band"]],
        df["claim_count"],
        constrained_var="age_band",
        increasing=False,
        family="poisson",
        link="log",
        exposure=df["exposure"],
    )
    assert res.n_obs == len(df)
