# Classification

Werner & Modlin §§9–10, GLM details from Goldburd et al. (2020).

## Two backends, one API

Both engines accept the same parameters; pick by name.

```python
from pyratemaking.glm import GLM

glm = GLM(family="poisson", link="log", backend="glum")          # default
glm = GLM(family="poisson", link="log", backend="statsmodels")   # exam-style output
```

`statsmodels` returns full inferential output (standard errors, p-values, deviance).
`glum` is faster, supports penalised regression, and treats Tweedie as native.

## Frequency-severity

```python
from pyratemaking.glm import FrequencySeverityModel

fs = FrequencySeverityModel.fit(
    X=policies[["region", "veh_brand", "driver_age"]],
    claim_count=policies["claim_count"],
    claim_amount=policies["incurred_losses"],
    exposure=policies["exposure"],
)
fs.relativities("region")
```

Frequency: Poisson with log-exposure offset.
Severity: Gamma weighted by claim count.
Pure premium: product of the two means.

## Tweedie

Single-model alternative when no clean split between frequency and severity is needed:

```python
from pyratemaking.glm import TweedieModel

model = TweedieModel.fit(
    X=policies[["region", "veh_brand"]],
    pure_premium=policies["pure_premium"],
    exposure=policies["exposure"],
    power=1.5,
)
```

## Penalised regression

Ridge, lasso, elastic net via `glum`:

```python
from pyratemaking.glm import fit_penalized

fit = fit_penalized(
    X, y,
    family="tweedie",
    penalty="elastic_net",
    alpha="cv",
    cv_alphas=(1e-4, 1e-3, 1e-2, 1e-1),
    sample_weight=exposure,
)
```

`alpha="cv"` runs k-fold CV on the supplied grid and picks the best.

## Stepwise selection

```python
from pyratemaking.glm import stepwise_select

sel = stepwise_select(
    X, y,
    candidates=["region", "veh_brand", "veh_gas", "driver_age"],
    direction="forward",
    criterion="aic",
    family="tweedie",
)
sel.selected
```

Forward, backward, or both, with AIC or BIC.

## Monotonicity constraints

Many rating variables are expected to be monotone in their level. Project the unconstrained relativities onto the monotone cone:

```python
from pyratemaking.glm import monotone_relativities, fit_monotone_glm

# Cheap post-hoc projection.
projected = monotone_relativities(rel_series, increasing=False)

# Iterative refit (close to projected gradient descent).
fit = fit_monotone_glm(X, y, constrained_var="vehicle_age", increasing=False)
```

## End-to-end via RatePlan

```python
plan.classify(rating_vars=["region", "veh_brand", "veh_gas"], family="tweedie", power=1.5)
plan.classification.relativities_frame()
```
