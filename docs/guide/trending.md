# Trending

Werner & Modlin §7. Project historical losses to the future rating period.

Three quantities trend separately: severity, frequency, and pure premium. Pure premium implicitly captures the product, but separating frequency and severity often improves the diagnostic.

## Multiplicative trend

```python
from pyratemaking.trending import severity_trend
import pandas as pd

losses = pd.Series([4.2e6, 4.4e6, 4.7e6, 5.0e6, 5.4e6], index=[2019, 2020, 2021, 2022, 2023])
counts = pd.Series([2_500, 2_500, 2_550, 2_600, 2_700], index=losses.index)

trend = severity_trend(losses, counts, kind="multiplicative")
print(f"Severity trend: {trend.annual_change:+.2%} per year")
print(trend.project([2024, 2025, 2026]))
```

The fit returns a `Trend` object with `predict`, `factor_to`, `confidence_interval`, and `project` methods. Confidence intervals come from the OLS standard error of the slope.

## Additive trend

For severity changes that look more like a constant dollar movement than a constant percentage:

```python
trend = severity_trend(losses, counts, kind="additive")
```

## Sensitivity analysis

Compare both forms across the candidate horizon to inform the trend selection:

```python
from pyratemaking.trending.core import sensitivity_table

sensitivity_table(losses / counts, times=losses.index, horizon=2026)
```

## Frequency and pure-premium trends

Same API:

```python
from pyratemaking.trending import frequency_trend, pure_premium_trend

frequency_trend(claim_count_by_year, exposure_by_year)
pure_premium_trend(losses_by_year, exposure_by_year)
```

The pure-premium trend should be approximately `(1 + freq_trend) × (1 + sev_trend) − 1` — a useful self-consistency check.
