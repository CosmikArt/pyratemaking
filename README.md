<p align="center">
  <strong>pyratemaking</strong><br>
  <em>Pure-Python ratemaking toolkit grounded in Werner &amp; Modlin.</em>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/pypi/v/pyratemaking?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-alpha-orange" alt="Status: Alpha"></a>
</p>

---

## What is pyratemaking?

**pyratemaking** is an opinionated Python library for property & casualty ratemaking, built around the methodology in *Basic Ratemaking* (Werner & Modlin, CAS, 5th ed.). It provides a single, composable API that takes you from raw loss triangles and exposure records to indicated rate changes — the workflow every pricing actuary runs quarterly but still cobbles together in Excel.

The actuarial Python ecosystem has solid reserving tools (`chainladder-python`) and general stats (`statsmodels`), but nothing that owns the **ratemaking** vertical end-to-end: exposure trending, on-leveling, GLM frequency–severity modeling, classification relativities, and diagnostics — all in one namespace. pyratemaking fills that gap.

## Installation

```bash
pip install pyratemaking
```

> **Note:** The package is not yet published on PyPI. For now, install from source:
> ```bash
> git clone https://github.com/CosmikArt/pyratemaking.git
> cd pyratemaking
> pip install -e ".[dev]"
> ```

## Quickstart

```python
import pyratemaking as prm
import pandas as pd

# --- 1. Load policy-level experience ---
experience = pd.DataFrame({
    "accident_year": [2020, 2020, 2021, 2021, 2022, 2022],
    "territory":     ["A", "B", "A", "B", "A", "B"],
    "earned_exposure": [1200, 800, 1350, 900, 1400, 950],
    "claim_count":   [60, 55, 58, 50, 65, 48],
    "incurred_loss": [480_000, 520_000, 470_000, 490_000, 530_000, 460_000],
})

# --- 2. Fit a frequency–severity GLM ---
model = prm.FrequencySeverityModel(
    freq_dist="poisson",
    sev_dist="gamma",
    link="log",
)
model.fit(
    experience,
    exposure_col="earned_exposure",
    freq_target="claim_count",
    sev_target="incurred_loss",
    features=["territory", "accident_year"],
)

# --- 3. Classification relativities ---
relativities = model.relativities(base_level={"territory": "A"})
print(relativities)
#   feature   level  relativity  std_error
# 0 territory   A      1.000      —
# 1 territory   B      0.923      0.041

# --- 4. Diagnostics ---
model.lift_chart(n_bins=10)            # observed vs predicted by decile
model.actual_vs_expected(by="territory")  # A/E ratio per class
```

## Features

| Module | Description | Status |
|--------|-------------|--------|
| `glm` | Frequency–severity GLMs (Poisson, Negative Binomial, Tweedie, Gamma) via `statsmodels` backend | 🚧 In progress |
| `exposure` | Earned/written exposure calculations, calendar-year and policy-year handling | 📋 Roadmap |
| `trending` | Loss and premium trending (exponential, linear) aligned to future effective period | 📋 Roadmap |
| `onlevel` | On-leveling of historical premiums using parallelogram method | 📋 Roadmap |
| `classification` | Classification ratemaking: minimum bias, Bailey–Simon, GLM-based relativities | 🚧 In progress |
| `baserate` | Indicated base rate and overall rate level change calculation | 📋 Roadmap |
| `diagnostics` | Lift charts, double lift charts, A/E ratios, residual plots, Gini coefficient | 🚧 In progress |
| `io` | Read/write ratemaking exhibits in CAS-standard format | 📋 Roadmap |

## API Design Principles

1. **DataFrames in, DataFrames out.** No custom container objects — everything flows through `pandas`.
2. **Werner & Modlin as spec.** Chapter references in docstrings so you can trace every formula.
3. **Composition over monoliths.** Each module works standalone; the `RatemakingPipeline` chains them.
4. **Diagnostics are first-class.** Every model ships with `.lift_chart()`, `.residuals()`, and `.actual_vs_expected()`.

## References

- Werner, G. & Modlin, C. (2016). *Basic Ratemaking*, 5th ed. Casualty Actuarial Society.
- De Jong, P. & Heller, G.Z. (2008). *Generalized Linear Models for Insurance Data*. Cambridge University Press.
- Goldburd, M., Khare, A. & Tevet, D. (2016). *Generalized Linear Models for Insurance Rating*, 2nd ed. CAS Monograph No. 5.
- Ohlsson, E. & Johansson, B. (2010). *Non-Life Insurance Pricing with Generalized Linear Models*. Springer.
- Anderson, D. et al. (2007). *A Practitioner's Guide to Generalized Linear Models*, 3rd ed. CAS Study Note.

## Contributing

Contributions welcome. This project follows a **docs-first** approach — if you want to add a module, start with the docstrings and a design note in `docs/`, then implement.

```bash
git clone https://github.com/CosmikArt/pyratemaking.git
cd pyratemaking
pip install -e ".[dev]"
pytest
```

Please open an issue before submitting large PRs.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Isaac López**
