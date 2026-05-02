# Large-loss procedures

Werner & Modlin §11.

## Basic limits

Cap individual claims at a stated limit so the data feeding indication is not driven by a few shock losses:

```python
from pyratemaking.large_loss import basic_limits_losses

out = basic_limits_losses(claims, basic_limit=250_000)
out["basic_limits_loss"]
out["excess_amount"]
out.attrs["summary"]   # per-AY totals + capped count
```

## Increased limits factors (ILFs)

Empirical ILFs from a censored loss sample:

```python
from pyratemaking.large_loss import increased_limits_factor_table

ilfs = increased_limits_factor_table(
    claims["claim_amount"],
    basic_limit=250_000,
    limits=[500_000, 1_000_000, 2_000_000, 5_000_000],
)
ilfs
```

The output is monotone increasing with limit and equals 1.0 at the basic limit by construction.

## Layer pricing

Empirical layer cost on a vector of losses:

```python
from pyratemaking.large_loss import layer_loss_cost

layer_loss_cost(claims["claim_amount"], attachment=250_000, limit=750_000)
```

With the optional `actudist` package, layers can be priced from a fitted severity distribution:

```python
import scipy.stats as st
from pyratemaking.large_loss import layer_pricing_from_distribution

layer_pricing_from_distribution(
    distribution=st.lognorm(s=1.0, scale=1e6),
    attachment=250_000,
    limit=750_000,
    n_samples=200_000,
)
```
