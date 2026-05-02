"""Policy table validation.

The downstream workflow expects one row per earned-exposure period with at
least an exposure column, an accident-year column, and (optionally) the
collected premium. Anything extra is preserved untouched so users can carry
their own rating variables through.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicySchema:
    """Column names for a policy table.

    Defaults match the canonical names used inside :class:`pyratemaking.RatePlan`,
    but every field can be overridden when the source data uses different names.
    """

    exposure: str = "exposure"
    ay: str = "policy_ay"
    premium: str | None = "earned_premium"
    policy_id: str | None = "policy_id"


def validate_policies(
    policies: pd.DataFrame,
    schema: PolicySchema | None = None,
    *,
    allow_zero_exposure: bool = True,
) -> pd.DataFrame:
    """Return a validated copy of ``policies`` ready for the pipeline.

    Parameters
    ----------
    policies : DataFrame
        Raw policy table.
    schema : PolicySchema, optional
        Column-name overrides. Defaults to the canonical names.
    allow_zero_exposure : bool, default True
        When False, rows with zero earned exposure are dropped (W&M §2 — zero
        exposure carries no information for ratemaking).

    Raises
    ------
    KeyError
        Required column missing.
    ValueError
        Negative exposure or premium values, or non-numeric AY column.
    """
    schema = schema or PolicySchema()
    out = policies.copy()

    _require_columns(out, [schema.exposure, schema.ay], context="policies")
    if schema.premium is not None and schema.premium not in out.columns:
        schema = PolicySchema(
            exposure=schema.exposure,
            ay=schema.ay,
            premium=None,
            policy_id=schema.policy_id,
        )

    exposure = pd.to_numeric(out[schema.exposure], errors="coerce")
    if exposure.isna().any():
        raise ValueError(f"non-numeric values in {schema.exposure!r}")
    if (exposure < 0).any():
        raise ValueError(f"negative values in {schema.exposure!r}")
    out[schema.exposure] = exposure.astype(float)

    ay = pd.to_numeric(out[schema.ay], errors="coerce")
    if ay.isna().any():
        raise ValueError(f"non-numeric values in {schema.ay!r}")
    out[schema.ay] = ay.astype(int)

    if schema.premium is not None:
        prem = pd.to_numeric(out[schema.premium], errors="coerce")
        if prem.isna().any():
            raise ValueError(f"non-numeric values in {schema.premium!r}")
        if (prem < 0).any():
            raise ValueError(f"negative values in {schema.premium!r}")
        out[schema.premium] = prem.astype(float)

    if not allow_zero_exposure:
        out = out.loc[out[schema.exposure] > 0].reset_index(drop=True)

    return out


def aggregate_to_ay(
    policies: pd.DataFrame,
    schema: PolicySchema | None = None,
) -> pd.DataFrame:
    """Sum exposure and premium by accident year.

    Returns a DataFrame indexed by AY with columns ``exposure`` and
    (when present) ``earned_premium``.
    """
    schema = schema or PolicySchema()
    cols = [schema.exposure]
    if schema.premium is not None and schema.premium in policies.columns:
        cols.append(schema.premium)

    grouped = policies.groupby(schema.ay, sort=True)[cols].sum()
    rename = {schema.exposure: "exposure"}
    if schema.premium is not None and schema.premium in policies.columns:
        rename[schema.premium] = "earned_premium"
    grouped = grouped.rename(columns=rename)
    grouped.index.name = "ay"
    return grouped


def _require_columns(df: pd.DataFrame, required: list[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{context} missing required columns: {missing}")


def attach_pure_premium(
    policies: pd.DataFrame,
    losses: pd.Series | np.ndarray,
    *,
    schema: PolicySchema | None = None,
    out_col: str = "pure_premium",
) -> pd.DataFrame:
    """Attach loss / exposure to the policy table as a pure-premium column."""
    schema = schema or PolicySchema()
    out = policies.copy()
    exposure = out[schema.exposure].to_numpy(dtype=float)
    losses_arr = np.asarray(losses, dtype=float)
    pp = np.where(exposure > 0, losses_arr / np.where(exposure > 0, exposure, 1.0), 0.0)
    out[out_col] = pp
    return out
