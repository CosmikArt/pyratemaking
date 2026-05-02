"""Overall rate level indication (W&M §8).

Two methods:

* **Loss-ratio** — `indicated change = (target_LR_components - current_LR) / current_LR`
  in its compact form, which is equivalent to comparing on-level pure-premium to
  the target after dividing both by exposure. Use when premium is reliable.

* **Pure-premium** — directly compares ultimate pure premium to current
  average rate. Use when on-leveling is impractical, e.g., new programs.

Both run through the *standard formula* of W&M Equation 8.2 (loss-ratio) and
8.3 (pure-premium):

    rate_change = (L + F) / (1 - V - Q) − 1                  (loss-ratio)
    indicated_rate = (PP + F_per_exposure) / (1 - V - Q)     (pure-premium)

with optional credibility weighting against an outside complement, per
W&M §12 / Mahler & Dean (2003).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ExpenseProvision:
    """Expense and profit provisions used by both indication methods."""

    fixed_expense_ratio: float = 0.0
    variable_expense_ratio: float = 0.0
    profit_and_contingency: float = 0.0
    other_acquisition: float = 0.0

    @property
    def variable_load(self) -> float:
        """Sum of provisions expressed as a fraction of premium."""
        return self.variable_expense_ratio + self.profit_and_contingency + self.other_acquisition

    def divisor(self) -> float:
        """``1 - V - Q`` denominator from W&M Eq. 8.2 / 8.3."""
        denom = 1.0 - self.variable_load
        if denom <= 0:
            raise ValueError("expense provisions exceed 1.0 — divisor is non-positive")
        return denom


@dataclass
class Indication:
    """Result of a rate-level indication run."""

    method: str
    target_loss_ratio: float | None
    indicated_rate_change: float
    credibility: float
    complement: float | None
    components: pd.Series
    contributions: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    def summary(self) -> pd.DataFrame:
        rows = [
            ("method", self.method),
            ("target_loss_ratio", self.target_loss_ratio),
            ("indicated_rate_change", self.indicated_rate_change),
            ("credibility", self.credibility),
            ("complement", self.complement),
        ]
        for k, v in self.components.items():
            rows.append((k, float(v)))
        return pd.DataFrame(rows, columns=["item", "value"]).set_index("item")

    def _repr_html_(self) -> str:  # pragma: no cover - exercised manually
        return self.summary().to_html()

    def __repr__(self) -> str:
        return (
            f"Indication(method={self.method!r}, "
            f"indicated_rate_change={self.indicated_rate_change:+.4%}, "
            f"credibility={self.credibility:.2f})"
        )


def loss_ratio_indication(
    on_level_premium: float | pd.Series,
    ultimate_losses: float | pd.Series,
    expenses: ExpenseProvision,
    *,
    target_loss_ratio: float | None = None,
    credibility: float = 1.0,
    complement: float = 0.0,
    weights: pd.Series | None = None,
) -> Indication:
    """Loss-ratio method (W&M Eq. 8.2).

    The target loss ratio defaults to ``1 - V - Q - F`` so the formula reduces
    to the standard form. Pass ``target_loss_ratio`` explicitly when working
    with a permissible loss ratio set by management.

    Pure indicated change before credibility:

        I = (L + F) / (1 - V - Q) - 1
          = experience_loss_ratio / target_loss_ratio - 1

    Credibility-weighted: ``Z * I + (1 - Z) * complement``.
    """
    L = _scalar_or_sum(ultimate_losses, on_level_premium, weights=weights, kind="losses")
    P = _scalar_or_sum(on_level_premium, None, weights=weights, kind="premium")
    if P <= 0:
        raise ValueError("on-level premium must be positive")
    experience_lr = L / P

    if target_loss_ratio is None:
        target_loss_ratio = max(expenses.divisor() - expenses.fixed_expense_ratio, 1e-12)

    raw_change = (experience_lr + expenses.fixed_expense_ratio) / expenses.divisor() - 1.0
    final_change = credibility * raw_change + (1.0 - credibility) * complement

    components = pd.Series(
        {
            "on_level_premium": P,
            "ultimate_losses": L,
            "experience_loss_ratio": experience_lr,
            "fixed_expense_ratio": expenses.fixed_expense_ratio,
            "variable_expense_ratio": expenses.variable_expense_ratio,
            "profit_and_contingency": expenses.profit_and_contingency,
            "raw_indicated_change": raw_change,
        }
    )
    return Indication(
        method="loss_ratio",
        target_loss_ratio=target_loss_ratio,
        indicated_rate_change=float(final_change),
        credibility=float(credibility),
        complement=float(complement),
        components=components,
    )


def pure_premium_indication(
    earned_exposure: float | pd.Series,
    ultimate_losses: float | pd.Series,
    expenses: ExpenseProvision,
    *,
    fixed_expense_per_exposure: float,
    current_average_rate: float,
    credibility: float = 1.0,
    complement: float = 0.0,
    weights: pd.Series | None = None,
) -> Indication:
    """Pure-premium method (W&M Eq. 8.3).

    Indicated average rate:

        R = (PP + F_per_exposure) / (1 - V - Q)

    Indicated rate change relative to current = ``R / current_avg_rate - 1``,
    optionally credibility-weighted against ``complement``.
    """
    L = _scalar_or_sum(ultimate_losses, earned_exposure, weights=weights, kind="losses")
    E = _scalar_or_sum(earned_exposure, None, weights=weights, kind="exposure")
    if E <= 0:
        raise ValueError("earned exposure must be positive")
    pure_premium = L / E
    indicated_rate = (pure_premium + fixed_expense_per_exposure) / expenses.divisor()
    if current_average_rate <= 0:
        raise ValueError("current_average_rate must be positive")
    raw_change = indicated_rate / current_average_rate - 1.0
    final_change = credibility * raw_change + (1.0 - credibility) * complement

    components = pd.Series(
        {
            "earned_exposure": E,
            "ultimate_losses": L,
            "pure_premium": pure_premium,
            "fixed_expense_per_exposure": fixed_expense_per_exposure,
            "indicated_avg_rate": indicated_rate,
            "current_avg_rate": current_average_rate,
            "raw_indicated_change": raw_change,
        }
    )
    return Indication(
        method="pure_premium",
        target_loss_ratio=None,
        indicated_rate_change=float(final_change),
        credibility=float(credibility),
        complement=float(complement),
        components=components,
    )


def _scalar_or_sum(
    value: float | pd.Series,
    aligned_with: pd.Series | None,
    *,
    weights: pd.Series | None,
    kind: str,
) -> float:
    """Reduce a Series (or scalar) to a single number with optional weights."""
    if isinstance(value, pd.Series):
        v = value.to_numpy(dtype=float)
        if weights is not None:
            w = weights.reindex(value.index).to_numpy(dtype=float)
            return float(np.nansum(v * w))
        return float(np.nansum(v))
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    raise TypeError(f"unexpected type for {kind}: {type(value).__name__}")
