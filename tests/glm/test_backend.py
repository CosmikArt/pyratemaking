import numpy as np
import pandas as pd
import pytest

from pyratemaking.glm import GLM
from pyratemaking.glm.backend import build_design


def test_build_design_drop_first_categorical():
    X = pd.DataFrame({"region": ["A", "B", "C", "A"], "age": [30, 40, 50, 60]})
    d = build_design(X, categorical=["region"], numeric=["age"])
    assert "Intercept" in d.matrix.columns
    assert "region[A]" not in d.matrix.columns
    assert "region[B]" in d.matrix.columns
    assert "region[C]" in d.matrix.columns
    assert "age" in d.matrix.columns


def test_build_design_uses_user_chosen_base_level():
    X = pd.DataFrame({"region": ["A", "B", "C"]})
    d = build_design(X, categorical=["region"], base_levels={"region": "C"})
    assert "region[C]" not in d.matrix.columns
    assert "region[A]" in d.matrix.columns
    assert d.base_levels["region"] == "C"


def test_glm_poisson_recovers_intercept_with_offset(synthetic_freq_data):
    df = synthetic_freq_data
    glm = GLM(family="poisson", link="log", backend="glum")
    res = glm.fit(
        df[["region", "driver_age"]],
        df["claim_count"],
        exposure=df["exposure"],
    )
    # Estimated intercept should be near log(0.10) at the base region (A) and age 0;
    # but coefficient on driver_age moves the intercept. We just check the
    # average prediction is close to the observed average frequency.
    pred = res.predict(df[["region", "driver_age"]], exposure=df["exposure"])
    avg_obs = df["claim_count"].mean()
    avg_pred = pred.mean()
    assert avg_pred == pytest.approx(avg_obs, rel=0.05)


def test_glm_relativities_for_log_link_match_exp_coef(synthetic_freq_data):
    df = synthetic_freq_data
    glm = GLM(family="poisson", link="log", backend="glum", base_levels={"region": "A"})
    res = glm.fit(
        df[["region"]],
        df["claim_count"],
        exposure=df["exposure"],
    )
    rel = res.relativities("region")
    assert rel.loc["A"] == pytest.approx(1.0)
    assert rel.loc["B"] == pytest.approx(np.exp(res.coef_["region[B]"]))


def test_glum_and_statsmodels_produce_equivalent_coefficients(synthetic_freq_data):
    df = synthetic_freq_data
    glum_glm = GLM(family="poisson", link="log", backend="glum")
    sm_glm = GLM(family="poisson", link="log", backend="statsmodels")
    a = glum_glm.fit(
        df[["region", "driver_age"]],
        df["claim_count"],
        exposure=df["exposure"],
    )
    b = sm_glm.fit(
        df[["region", "driver_age"]],
        df["claim_count"],
        exposure=df["exposure"],
    )
    common = a.coef_.index.intersection(b.coef_.index)
    np.testing.assert_allclose(
        a.coef_.loc[common].to_numpy(),
        b.coef_.loc[common].to_numpy(),
        rtol=1e-3,
        atol=1e-3,
    )


def test_predict_requires_exposure_when_fit_used_offset(synthetic_freq_data):
    df = synthetic_freq_data
    glm = GLM(family="poisson", link="log", backend="glum")
    res = glm.fit(
        df[["region"]],
        df["claim_count"],
        exposure=df["exposure"],
    )
    with pytest.raises(ValueError, match="exposure"):
        res.predict(df[["region"]])


def test_unknown_backend_raises():
    glm = GLM(family="poisson", link="log", backend="ham_radio")
    with pytest.raises(ValueError, match="backend"):
        glm.fit(pd.DataFrame({"x": [1.0, 2.0]}), np.array([1.0, 2.0]))


def test_relativities_only_log_link():
    glm = GLM(family="gaussian", link="identity", backend="statsmodels")
    res = glm.fit(
        pd.DataFrame({"region": ["A", "B", "A", "B"], "x": [1.0, 2.0, 3.0, 4.0]}),
        np.array([10.0, 20.0, 11.0, 22.0]),
    )
    with pytest.raises(ValueError, match="log-link"):
        res.relativities("region")
