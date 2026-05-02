# On-leveling

Werner & Modlin §5. Brings historical premium to the level it would have been at under today's rates.

## Parallelogram method

Geometric approximation under uniform writing. Cheap, schedule-only.

```python
from pyratemaking.onleveling import RateChange, parallelogram
import pandas as pd

ep = pd.Series([1_200_000, 1_300_000, 1_400_000], index=[2021, 2022, 2023])
changes = [RateChange(date=2021.5, factor=1.05), RateChange(date=2022.75, factor=1.03)]
out = parallelogram(ep, changes)
out
```

The output frame has `avg_rate_level`, `on_level_factor`, and `on_level_premium` per AY.

The geometry assumes:

- Annual policies, uniformly written across the calendar year.
- Identical policy term across the historical period.
- No mid-term endorsements moving the rate level.

## Extension of exposures

Re-rate every policy using today's rating algorithm. Captures rating-variable mix shifts that the parallelogram method cannot.

```python
from pyratemaking.onleveling import extension_of_exposures

def algorithm(df):
    base = 1_000.0
    territory_factor = df["territory"].map({"A": 1.0, "B": 1.10, "C": 0.90})
    return base * territory_factor.to_numpy()

out = extension_of_exposures(policies, algorithm)
```

The algorithm is a callable receiving the full DataFrame and returning a per-row premium array. The function does not assume any specific factor structure — that's the user's domain knowledge.

## When to choose which

| Situation | Use |
|---|---|
| Rate changes only, no segment mix shift | Parallelogram |
| Significant mix shift between writing periods | Extension of exposures |
| Detailed rating algorithm available, full policy table | Extension of exposures |
| Quick exhibit on aggregate premium | Parallelogram |
