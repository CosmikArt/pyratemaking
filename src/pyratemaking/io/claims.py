"""Claims table validation and aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClaimsSchema:
    """Column names for a claims table.

    A claim is identified by its accident year and (when present) a policy id
    that ties it back to the policy table. Loss amounts are required.
    """

    loss: str = "claim_amount"
    ay: str = "policy_ay"
    policy_id: str | None = "policy_id"
    report_lag: str | None = None


def validate_claims(
    claims: pd.DataFrame,
    schema: ClaimsSchema | None = None,
    *,
    allow_zero_loss: bool = True,
) -> pd.DataFrame:
    """Return a validated copy of ``claims`` ready for the pipeline."""
    schema = schema or ClaimsSchema()
    out = claims.copy()

    _require_columns(out, [schema.loss, schema.ay], context="claims")

    losses = pd.to_numeric(out[schema.loss], errors="coerce")
    if losses.isna().any():
        raise ValueError(f"non-numeric values in {schema.loss!r}")
    if (losses < 0).any():
        raise ValueError(f"negative values in {schema.loss!r}")
    out[schema.loss] = losses.astype(float)

    ay = pd.to_numeric(out[schema.ay], errors="coerce")
    if ay.isna().any():
        raise ValueError(f"non-numeric values in {schema.ay!r}")
    out[schema.ay] = ay.astype(int)

    if not allow_zero_loss:
        out = out.loc[out[schema.loss] > 0].reset_index(drop=True)

    return out


def aggregate_to_ay(
    claims: pd.DataFrame,
    schema: ClaimsSchema | None = None,
) -> pd.DataFrame:
    """Sum incurred losses and claim counts by accident year.

    Returns a frame indexed by AY with columns ``incurred_losses`` and
    ``claim_count`` (count of claims with positive amount).
    """
    schema = schema or ClaimsSchema()
    grouped = claims.groupby(schema.ay, sort=True).agg(
        incurred_losses=(schema.loss, "sum"),
        claim_count=(schema.loss, lambda s: int((s > 0).sum())),
    )
    grouped.index.name = "ay"
    return grouped


def merge_policy_losses(
    policies: pd.DataFrame,
    claims: pd.DataFrame,
    *,
    policy_schema_id: str = "policy_id",
    claims_schema: ClaimsSchema | None = None,
) -> pd.DataFrame:
    """Attach total incurred loss per policy to the policy table.

    Used when fitting policy-level severity / pure-premium GLMs.
    """
    claims_schema = claims_schema or ClaimsSchema()
    if policy_schema_id not in policies.columns:
        raise KeyError(f"policies missing {policy_schema_id!r}")
    if claims_schema.policy_id is None or claims_schema.policy_id not in claims.columns:
        raise KeyError("claims schema must declare a policy_id column for merging")

    by_policy = (
        claims.groupby(claims_schema.policy_id)[claims_schema.loss]
        .agg(["sum", "count"])
        .rename(columns={"sum": "incurred_losses", "count": "claim_count"})
    )
    base = policies.drop(
        columns=[c for c in ("incurred_losses", "claim_count") if c in policies.columns]
    )
    out = base.merge(
        by_policy,
        how="left",
        left_on=policy_schema_id,
        right_index=True,
    )
    out["incurred_losses"] = out["incurred_losses"].fillna(0.0).astype(float)
    out["claim_count"] = out["claim_count"].fillna(0).astype(int)
    return out


def _require_columns(df: pd.DataFrame, required: list[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{context} missing required columns: {missing}")


def loss_triangle(
    claims: pd.DataFrame,
    *,
    ay_col: str = "policy_ay",
    dev_col: str = "development_age",
    loss_col: str = "claim_amount",
    cumulative: bool = True,
) -> pd.DataFrame:
    """Build an AY × dev-age loss triangle.

    Parameters
    ----------
    claims : DataFrame
        Long-format claims table with one row per AY × dev-age cell, or one
        row per claim with a development age column.
    cumulative : bool, default True
        Return cumulative losses. When False, returns incremental.

    Returns
    -------
    DataFrame indexed by AY, columns are integer development ages, values are
    paid or incurred losses (whichever is in ``loss_col``).
    """
    if dev_col not in claims.columns:
        raise KeyError(f"loss triangle requires {dev_col!r} in claims")

    pivot = claims.pivot_table(
        index=ay_col,
        columns=dev_col,
        values=loss_col,
        aggfunc="sum",
        fill_value=np.nan,
    )
    pivot = pivot.sort_index().sort_index(axis=1)

    if cumulative:
        with np.errstate(invalid="ignore"):
            return pivot.cumsum(axis=1).where(pivot.notna())
    return pivot
