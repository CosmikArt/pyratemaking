import pandas as pd
import pytest

from pyratemaking.core.indication import (
    ExpenseProvision,
    loss_ratio_indication,
    pure_premium_indication,
)


def test_loss_ratio_indication_matches_textbook_formula():
    # W&M Eq. 8.2 example: L=0.65, F=0.05, V=0.25, Q=0.05.
    # Required change = (0.65 + 0.05) / (1 - 0.25 - 0.05) - 1 = 0.7/0.7 - 1 = 0.
    expenses = ExpenseProvision(
        fixed_expense_ratio=0.05,
        variable_expense_ratio=0.25,
        profit_and_contingency=0.05,
    )
    out = loss_ratio_indication(
        on_level_premium=10_000_000,
        ultimate_losses=6_500_000,
        expenses=expenses,
    )
    assert out.indicated_rate_change == pytest.approx(0.0, abs=1e-12)


def test_loss_ratio_indication_positive_change_when_lr_high():
    expenses = ExpenseProvision(
        fixed_expense_ratio=0.05,
        variable_expense_ratio=0.25,
        profit_and_contingency=0.05,
    )
    out = loss_ratio_indication(
        on_level_premium=10_000_000,
        ultimate_losses=8_000_000,
        expenses=expenses,
    )
    expected = (0.80 + 0.05) / 0.70 - 1
    assert out.indicated_rate_change == pytest.approx(expected, rel=1e-12)


def test_loss_ratio_credibility_blends_with_complement():
    expenses = ExpenseProvision(variable_expense_ratio=0.25, profit_and_contingency=0.05)
    out = loss_ratio_indication(
        on_level_premium=10_000_000,
        ultimate_losses=8_000_000,
        expenses=expenses,
        credibility=0.5,
        complement=0.0,
    )
    raw = (0.80) / 0.70 - 1
    assert out.indicated_rate_change == pytest.approx(0.5 * raw, rel=1e-12)


def test_loss_ratio_zero_premium_raises():
    expenses = ExpenseProvision(variable_expense_ratio=0.25)
    with pytest.raises(ValueError, match="premium must be positive"):
        loss_ratio_indication(0.0, 100_000, expenses)


def test_pure_premium_indication_matches_textbook_formula():
    # PP = 200, F_per_exposure = 50, V=0.25, Q=0.05; divisor = 0.70.
    # Indicated rate = (200 + 50) / 0.70 = 357.14
    # Current rate = 350 → change = 357.14/350 - 1 = 2.04%
    expenses = ExpenseProvision(variable_expense_ratio=0.25, profit_and_contingency=0.05)
    out = pure_premium_indication(
        earned_exposure=10_000,
        ultimate_losses=2_000_000,
        expenses=expenses,
        fixed_expense_per_exposure=50.0,
        current_average_rate=350.0,
    )
    expected = (200 + 50) / 0.70 / 350 - 1
    assert out.indicated_rate_change == pytest.approx(expected, rel=1e-12)


def test_pure_premium_with_series_inputs_aggregates_correctly():
    expenses = ExpenseProvision(variable_expense_ratio=0.25)
    exposure = pd.Series({2020: 5000, 2021: 5500, 2022: 6000})
    losses = pd.Series({2020: 1_000_000, 2021: 1_100_000, 2022: 1_200_000})
    out = pure_premium_indication(
        exposure,
        losses,
        expenses,
        fixed_expense_per_exposure=20.0,
        current_average_rate=300.0,
    )
    pp = losses.sum() / exposure.sum()
    expected = (pp + 20.0) / 0.75 / 300.0 - 1
    assert out.indicated_rate_change == pytest.approx(expected, rel=1e-12)


def test_expense_provision_divisor_blocks_negative():
    with pytest.raises(ValueError, match="non-positive"):
        ExpenseProvision(variable_expense_ratio=0.95, profit_and_contingency=0.10).divisor()


def test_summary_lists_all_components():
    expenses = ExpenseProvision(variable_expense_ratio=0.25)
    out = loss_ratio_indication(1_000_000, 600_000, expenses)
    s = out.summary()
    assert "experience_loss_ratio" in s.index
    assert "indicated_rate_change" in s.index
