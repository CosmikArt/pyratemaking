# Implementation

Werner & Modlin §14. Once an indicated rate change is decided, applying it uniformly is rare. Caps and floors limit the per-policy impact; dispersion reports show how the new rates fall across the existing book.

## Caps and floors

```python
from pyratemaking.core.implementation import apply_caps_floors

capped = apply_caps_floors(
    current_premium=policies["current_premium"],
    indicated_premium=policies["indicated_premium"],
    cap=1.15,    # at most +15% per policy
    floor=0.85,  # at most -15% per policy
)
```

## Per-policy impact

```python
from pyratemaking.core.implementation import implement_rate_change

result = implement_rate_change(
    policies=policies,
    current_premium_col="current_premium",
    indicated_premium_col="indicated_premium",
    cap=1.15,
    floor=0.85,
    extra_columns=["region", "veh_brand"],
)
```

The returned `ImplementationResult` carries:

- `impacted` — per-policy frame with `pct_change`, `was_capped`, `was_floored`.
- `dispersion_summary()` — share of book by rate-change bucket.
- `segment_summary("region")` — average and median change by segment.
- `share_above_threshold(0.05)` / `share_below_threshold(-0.05)` — quick sanity numbers.

## Via RatePlan

```python
plan.implement(cap=1.15, floor=0.85)
plan.implementation.dispersion_summary()
plan.implementation.segment_summary("region")
```

## Choosing caps and floors

Common practice: cap at the same magnitude as the overall indication (e.g. +15% with a 13.5% indication) to avoid concentrated dislocations. Floors are typically symmetric, but a no-floor implementation is also common for renewal books where retention is sensitive to surprises.
