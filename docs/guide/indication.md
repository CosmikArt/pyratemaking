# Rate level indication

Werner & Modlin §8 — the overall rate level indication answers a single question: by what percentage should the average premium move?

## Loss-ratio method (W&M Eq. 8.2)

$$
\text{indicated change} = \frac{L + F}{1 - V - Q} - 1
$$

with $L$ the experience loss ratio (ultimate losses ÷ on-level premium), $F$ the fixed expense ratio, $V$ the variable expense ratio, and $Q$ the profit and contingency provision.

Use this when you have reliable on-level premium and a sensible expense breakdown.

```python
from pyratemaking.core.indication import ExpenseProvision, loss_ratio_indication

expenses = ExpenseProvision(
    fixed_expense_ratio=0.05,
    variable_expense_ratio=0.20,
    profit_and_contingency=0.05,
)
out = loss_ratio_indication(
    on_level_premium=12_500_000,
    ultimate_losses=8_400_000,
    expenses=expenses,
)
print(out.indicated_rate_change)
```

## Pure-premium method (W&M Eq. 8.3)

$$
R = \frac{PP + F_{\text{per exposure}}}{1 - V - Q}
$$

Indicated change against the current average rate. Use this for new programs where on-leveling is impractical, or when premium is unreliable.

```python
from pyratemaking.core.indication import pure_premium_indication

out = pure_premium_indication(
    earned_exposure=50_000,
    ultimate_losses=10_000_000,
    expenses=expenses,
    fixed_expense_per_exposure=15.0,
    current_average_rate=240.0,
)
```

## Credibility weighting

Both methods accept `credibility` and `complement`. The weighted indication is

$$
Z \cdot I_{\text{raw}} + (1 - Z) \cdot \text{complement}
$$

where the complement is typically the countrywide indication or a published trend.

## Common pitfalls

- Using collected (rather than on-level) premium understates the change when rates have already moved.
- Mixing accident-year losses with calendar-year premium without aligning earning periods.
- Omitting LAE from the loss provision when LAE is built into the expense ratio (double counting).
