import numpy as np
import pytest

from pyratemaking.glm import TweedieModel


def test_tweedie_fit_log_link_yields_positive_predictions(synthetic_pure_premium_data):
    df = synthetic_pure_premium_data
    model = TweedieModel.fit(
        df[["region", "driver_age"]],
        df["pure_premium"],
        exposure=df["exposure"],
        power=1.5,
    )
    pred = model.predict(df[["region", "driver_age"]])
    assert (pred > 0).all()


def test_tweedie_relativities_for_region(synthetic_pure_premium_data):
    df = synthetic_pure_premium_data
    model = TweedieModel.fit(
        df[["region"]],
        df["pure_premium"],
        exposure=df["exposure"],
        power=1.5,
        base_levels={"region": "A"},
    )
    rel = model.relativities("region")
    assert rel.loc["A"] == pytest.approx(1.0)
    assert (rel > 0).all()
