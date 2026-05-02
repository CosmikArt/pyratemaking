# pyratemaking 📊

**Pure-Python ratemaking for P&C insurance — end to end.**

[![PyPI](https://img.shields.io/pypi/v/pyratemaking.svg)](https://pypi.org/project/pyratemaking/)
[![Python](https://img.shields.io/pypi/pyversions/pyratemaking.svg)](https://pypi.org/project/pyratemaking/)
[![Tests](https://github.com/CosmikArt/pyratemaking/actions/workflows/test.yml/badge.svg)](https://github.com/CosmikArt/pyratemaking/actions/workflows/test.yml)
[![Coverage](https://codecov.io/gh/CosmikArt/pyratemaking/branch/main/graph/badge.svg)](https://codecov.io/gh/CosmikArt/pyratemaking)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`pyratemaking` is the Python implementation of the Werner & Modlin ratemaking workflow: rate level indication, on-leveling, trending, development, classification analysis, large loss procedures, and rate implementation. Built for P&C actuaries who want a reproducible pipeline from raw policies and claims to a defensible rate filing.

## Why

Python has GLM libraries. It has chain-ladder libraries. It does not have an end-to-end ratemaking workflow. `pyratemaking` is that workflow.

## Install

```bash
pip install pyratemaking            # core
pip install pyratemaking[full]      # with full ecosystem (whsmooth, actudist, etc.)
```

## Quickstart

```python
from pyratemaking import RatePlan
from pyratemaking.datasets import french_motor

policies, claims = french_motor.load()

plan = RatePlan(
    policies=policies,
    claims=claims,
    exposure_col="exposure",
    loss_col="claim_amount",
    ay_col="policy_ay",
)

indication = plan.indicate(method="loss_ratio", target_lr=0.65)
print(indication.summary())

plan.classify(
    rating_vars=["driver_age", "veh_power", "region"],
    backend="glum",
    family="tweedie",
    power=1.5,
)

plan.implement(cap=1.15, floor=0.85)
plan.report.filing("filing_2026.html")
```

## Werner & Modlin coverage

| Chapter | Topic                          | Module                            |
| ------- | ------------------------------ | --------------------------------- |
| 5       | On-leveling                    | `pyratemaking.onleveling`         |
| 6       | Loss development               | `pyratemaking.development`        |
| 7       | Trending                       | `pyratemaking.trending`           |
| 8       | Overall rate level indication  | `pyratemaking.core.indication`    |
| 9-10    | Classification ratemaking      | `pyratemaking.core.classification`|
| 11      | Increased limits / large loss  | `pyratemaking.large_loss`         |
| 12      | GLMs                           | `pyratemaking.glm`                |
| 14      | Implementation                 | `pyratemaking.core.implementation`|

## Ecosystem

Part of a 6-library actuarial Python suite:

- [`actudist`](https://pypi.org/project/actudist/) — severity and frequency distributions
- [`burncost`](https://pypi.org/project/burncost/) — burning cost analysis
- [`actuarcredibility`](https://pypi.org/project/actuarcredibility/) — credibility methods
- [`whsmooth`](https://pypi.org/project/whsmooth/) — Whittaker-Henderson smoothing
- **`pyratemaking`** — ratemaking workflow (this library)
- `pyinsurancerating` — rating engine (coming)

## Roadmap

Future releases will add Bayesian hierarchical credibility, deep-learning rating models (LocalGLMnet, CANN), premium calculation principles, fairness auditing, a CLI tool, and a Streamlit exploration dashboard. See [milestones](https://github.com/CosmikArt/pyratemaking/milestones).

## References

- Werner, G. & Modlin, C. (2016). *Basic Ratemaking* (5th ed.). Casualty Actuarial Society.
- Goldburd, M., Khare, A., Tevet, D. & Guller, D. (2020). *Generalized Linear Models for Insurance Rating* (2nd ed.). CAS Monograph 5.
- Friedland, J. (2013). *Fundamentals of General Insurance Actuarial Analysis*. Society of Actuaries.
- Mack, T. (1993). "Distribution-free Calculation of the Standard Error of Chain Ladder Reserve Estimates." *ASTIN Bulletin*, 23(2), 213-225.

## License

MIT © Isaac López
