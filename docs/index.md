# pyratemaking

Pure-Python ratemaking for P&C insurance, end to end.

`pyratemaking` implements the workflow from Werner & Modlin (2016), *Basic Ratemaking* (CAS, 5th ed.): rate level indication, on-leveling, trending, loss development, classification analysis, large-loss procedures, and rate implementation. Built for P&C actuaries who want a reproducible pipeline from raw policies and claims to a defensible rate filing.

## Why

Python has GLM libraries. It has chain-ladder libraries. It does not have an end-to-end ratemaking workflow that owns the whole indication-to-implementation arc. `pyratemaking` is that workflow.

## Install

```bash
pip install pyratemaking            # core
pip install "pyratemaking[full]"    # with the rest of the actuarial ecosystem
```

## At a glance

```python
from pyratemaking import RatePlan
from pyratemaking.datasets import synthetic

policies, claims = synthetic.generate(n_policies=10_000, seed=42)

plan = RatePlan(policies=policies, claims=claims)
plan.indicate(method="loss_ratio")
plan.classify(rating_vars=["region", "veh_brand", "veh_gas"], family="tweedie", power=1.5)
plan.implement(cap=1.15, floor=0.85)

plan.report.filing("filing.html")
```

## Werner & Modlin coverage

| Chapter | Topic                          | Module                            |
| ------- | ------------------------------ | --------------------------------- |
| 5       | On-leveling                    | `pyratemaking.onleveling`         |
| 6       | Loss development               | `pyratemaking.development`        |
| 7       | Trending                       | `pyratemaking.trending`           |
| 8       | Overall rate level indication  | `pyratemaking.core.indication`    |
| 9–10    | Classification ratemaking      | `pyratemaking.core.classification`|
| 11      | Increased limits / large loss  | `pyratemaking.large_loss`         |
| 12      | GLMs                           | `pyratemaking.glm`                |
| 14      | Implementation                 | `pyratemaking.core.implementation`|

## References

- Werner, G. & Modlin, C. (2016). *Basic Ratemaking* (5th ed.). Casualty Actuarial Society.
- Goldburd, M., Khare, A., Tevet, D. & Guller, D. (2020). *Generalized Linear Models for Insurance Rating* (2nd ed.). CAS Monograph 5.
- Mack, T. (1993). "Distribution-free Calculation of the Standard Error of Chain Ladder Reserve Estimates." *ASTIN Bulletin*, 23(2), 213–225.
- Friedland, J. (2013). *Fundamentals of General Insurance Actuarial Analysis*. Society of Actuaries.

## License

MIT.
