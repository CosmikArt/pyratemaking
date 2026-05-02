"""Synthetic auto-insurance dataset.

Generates plausible policies and claims with the same column schema as
:mod:`french_motor`, so any pipeline that runs on French Motor runs here too.
The simulation uses a U-shaped driver-age curve, a vehicle-power frequency
gradient, a region effect, and a Gamma severity model with frequency-severity
correlation. Deterministic given ``seed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_AGE_BINS = np.array([18, 25, 30, 40, 50, 60, 70, 95])
_AGE_FREQ_RELATIVITY = np.array([1.55, 1.20, 1.00, 0.95, 0.92, 0.95, 1.10])


def _age_relativity(age: np.ndarray) -> np.ndarray:
    idx = np.clip(
        np.searchsorted(_AGE_BINS, age, side="right") - 1, 0, len(_AGE_FREQ_RELATIVITY) - 1
    )
    return _AGE_FREQ_RELATIVITY[idx]


def generate(
    n_policies: int = 10_000,
    *,
    seed: int = 42,
    base_frequency: float = 0.10,
    base_severity: float = 1500.0,
    severity_dispersion: float = 0.7,
    region_effect: dict[str, float] | None = None,
    veh_brand_effect: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate ``(policies, claims)`` with French-Motor-compatible schema.

    Parameters
    ----------
    n_policies : int, default 10_000
    seed : int, default 42
    base_frequency : float, default 0.10
        Base annual claim frequency at age 35, region "R24", brand "B1".
    base_severity : float, default 1500.0
        Base average severity in the same conditions.
    severity_dispersion : float, default 0.7
        Gamma shape parameter (smaller = heavier-tailed).
    region_effect, veh_brand_effect : dict, optional
        Multiplicative log-effects on frequency. Defaults are realistic.
    """
    rng = np.random.default_rng(seed)
    region_effect = region_effect or {
        "R24": 0.0,
        "R52": 0.10,
        "R93": -0.10,
        "R11": 0.05,
        "R72": -0.05,
    }
    veh_brand_effect = veh_brand_effect or {
        "B1": 0.0,
        "B2": 0.05,
        "B12": 0.10,
        "B3": -0.05,
        "B5": 0.0,
    }
    region = rng.choice(list(region_effect.keys()), size=n_policies)
    veh_brand = rng.choice(list(veh_brand_effect.keys()), size=n_policies)
    veh_gas = rng.choice(["Diesel", "Regular"], size=n_policies, p=[0.55, 0.45])
    area = rng.choice(["A", "B", "C", "D", "E", "F"], size=n_policies)
    driver_age = rng.integers(18, 95, size=n_policies)
    veh_age = rng.integers(0, 30, size=n_policies)
    veh_power = rng.integers(4, 16, size=n_policies)
    bonus_malus = np.clip(rng.normal(70, 25, size=n_policies), 50, 230).astype(int)
    density = np.clip(rng.lognormal(mean=4.5, sigma=1.0, size=n_policies), 1, 30_000).astype(int)
    exposure = np.clip(rng.uniform(0.05, 1.0, size=n_policies), 0.05, 1.0)

    age_factor = _age_relativity(driver_age)
    region_factor = np.exp(np.array([region_effect[r] for r in region]))
    brand_factor = np.exp(np.array([veh_brand_effect[b] for b in veh_brand]))
    bm_factor = (bonus_malus / 100.0) ** 1.2
    power_factor = (veh_power / 6.0) ** 0.4

    log_lambda = (
        np.log(base_frequency)
        + np.log(age_factor)
        + np.log(region_factor)
        + np.log(brand_factor)
        + np.log(bm_factor)
        + np.log(power_factor)
    )
    expected_claims = np.exp(log_lambda) * exposure
    claim_count = rng.poisson(expected_claims)

    policy_id = np.arange(1, n_policies + 1)
    policies = pd.DataFrame(
        {
            "policy_id": policy_id,
            "policy_ay": 2024,
            "exposure": exposure,
            "earned_premium": np.exp(log_lambda) * base_severity * exposure / 0.65,
            "area": area,
            "veh_power": veh_power,
            "veh_age": veh_age,
            "driver_age": driver_age,
            "bonus_malus": bonus_malus,
            "veh_brand": veh_brand,
            "veh_gas": veh_gas,
            "density": density,
            "region": region,
            "claim_count": claim_count,
        }
    )

    claim_rows = []
    for pid, n_claims, lam in zip(policy_id, claim_count, np.exp(log_lambda), strict=False):
        for _ in range(int(n_claims)):
            mean_severity = base_severity * (lam / base_frequency) ** 0.4
            shape = severity_dispersion
            scale = mean_severity / shape
            amount = float(rng.gamma(shape=shape, scale=scale))
            claim_rows.append(
                {
                    "policy_id": int(pid),
                    "policy_ay": 2024,
                    "claim_amount": amount,
                }
            )
    claims = pd.DataFrame(
        claim_rows,
        columns=["policy_id", "policy_ay", "claim_amount"],
    )
    return policies, claims
