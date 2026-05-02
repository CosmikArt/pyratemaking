# Quickstart

Five steps from raw data to a rate filing.

## 1. Load data

`pyratemaking` expects two tables: a per-policy frame and a per-claim frame. The synthetic generator below mirrors the schema of the French Motor TPL dataset.

```python
from pyratemaking.datasets import synthetic

policies, claims = synthetic.generate(n_policies=20_000, seed=42)
```

The policy table needs at least an exposure column and an accident-year column. The claims table needs a loss column and an accident-year column. Defaults match the synthetic and French Motor schemas.

## 2. Build the plan

```python
from pyratemaking import RatePlan

plan = RatePlan(policies=policies, claims=claims)
```

`RatePlan` validates the input, attaches losses to policies, and prepares helpers for indication, classification, implementation, diagnostics, and reporting.

## 3. Indicate

```python
indication = plan.indicate(method="loss_ratio")
print(indication.summary())
```

Pass `target_lr=...`, `expenses=ExpenseProvision(...)`, `credibility=...`, and `complement=...` to override the defaults. `method="pure_premium"` uses the alternative form (W&M Eq. 8.3).

## 4. Classify

```python
plan.classify(
    rating_vars=["region", "veh_brand", "veh_gas"],
    family="tweedie",
    backend="glum",
    power=1.5,
)
```

Choose `backend="statsmodels"` for the classic exam-comparable output. Add `penalty="elastic_net", alpha="cv"` to fit a penalised GLM with cross-validated alpha (Tweedie path uses `glum` either way).

## 5. Implement and report

```python
plan.implement(cap=1.15, floor=0.85)
plan.report.filing("filing.html")
plan.report.excel("filing.xlsx")
```

The HTML filing is a single self-contained file with every exhibit; the Excel workbook has one sheet per exhibit, formatted for review.

## Diagnostics

```python
plan.diagnostics.lift(n_bins=10).figure().savefig("lift.png")
plan.diagnostics.gini()
plan.diagnostics.actual_vs_expected(by="region")
```

See the user guide for individual modules and the API reference for full signatures.
