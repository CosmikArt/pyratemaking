import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_freq_data():
    rng = np.random.default_rng(0)
    n = 20_000
    region = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    age = rng.integers(20, 70, size=n)
    exposure = rng.uniform(0.2, 1.0, size=n)
    region_effect = pd.Series({"A": 0.0, "B": 0.30, "C": -0.20})
    age_effect = -0.01 * (age - 40)
    log_lambda = np.log(0.10) + region_effect.loc[region].to_numpy() + age_effect
    counts = rng.poisson(np.exp(log_lambda) * exposure)
    df = pd.DataFrame(
        {
            "region": region,
            "driver_age": age.astype(float),
            "exposure": exposure,
            "claim_count": counts,
        }
    )
    return df


@pytest.fixture
def synthetic_pure_premium_data():
    rng = np.random.default_rng(7)
    n = 12_000
    region = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    age = rng.integers(20, 70, size=n).astype(float)
    exposure = rng.uniform(0.2, 1.0, size=n)
    region_effect = pd.Series({"A": 0.0, "B": 0.20, "C": -0.10})
    age_effect = -0.01 * (age - 40)
    log_mu = np.log(800.0) + region_effect.loc[region].to_numpy() + age_effect
    pp = rng.gamma(shape=0.4, scale=np.exp(log_mu) / 0.4) * (rng.random(n) > 0.7)
    df = pd.DataFrame(
        {
            "region": region,
            "driver_age": age,
            "exposure": exposure,
            "pure_premium": pp,
        }
    )
    return df
