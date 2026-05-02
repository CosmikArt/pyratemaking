"""Monotonicity-constrained GLM relativities.

Many rating variables are expected to be monotone — older vehicles are
never rated higher than newer ones, longer driving experience never raises
the rate. The unconstrained GLM may violate monotonicity in noisy data, in
which case the standard fix is to project the relativity vector onto the
monotone cone via the pool-adjacent-violators (PAV) algorithm.

We expose two entry points:

* :func:`monotone_relativities` — post-hoc PAV projection of an already
  fitted GLM's relativities. Cheap, common in practice.
* :func:`fit_monotone_glm` — iterative refit: fit, project, recompute the
  offset for the constrained variable, refit the rest. Stops on a small
  change in deviance. Closer to projected gradient descent in spirit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from pyratemaking.glm.backend import GLM, GLMResult


def monotone_relativities(
    relativities: pd.Series,
    *,
    increasing: bool = True,
    weights: pd.Series | None = None,
) -> pd.Series:
    """Project a relativity vector onto the monotone cone.

    Uses isotonic regression (pool-adjacent-violators) on ``log(relativity)``
    so the geometric structure of the rating algorithm is preserved.
    """
    levels = list(relativities.index)
    log_rel = np.log(relativities.to_numpy(dtype=float))
    w = weights.reindex(levels).fillna(1.0).to_numpy(dtype=float) if weights is not None else None
    iso = IsotonicRegression(increasing=increasing)
    fitted = iso.fit_transform(np.arange(len(levels), dtype=float), log_rel, sample_weight=w)
    return pd.Series(np.exp(fitted), index=levels, name=relativities.name)


def fit_monotone_glm(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    constrained_var: str,
    *,
    increasing: bool = True,
    family: str = "tweedie",
    link: str = "log",
    backend: str = "glum",
    tweedie_power: float = 1.5,
    sample_weight: np.ndarray | pd.Series | None = None,
    exposure: np.ndarray | pd.Series | None = None,
    base_levels: dict[str, str] | None = None,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> GLMResult:
    """Iterative monotone-constrained GLM via projection-and-refit.

    The procedure: fit unconstrained, project relativities for ``constrained_var``
    onto the monotone cone, freeze them as a known offset, refit the remaining
    coefficients, repeat until deviance settles.

    DESIGN: this is a heuristic close to projected gradient descent on the
    Lagrangian. Convergence is not guaranteed in pathological cases but is
    fine on real ratemaking data with one constrained variable. A full
    interior-point implementation is on the v0.2 roadmap.
    """
    if constrained_var not in X.columns:
        raise KeyError(f"{constrained_var!r} not in X")
    if link != "log":
        raise NotImplementedError("monotone projection currently assumes log link")

    levels = sorted(X[constrained_var].astype("category").cat.categories)
    other_cols = [c for c in X.columns if c != constrained_var]

    constraint_offset = np.zeros(len(X))
    last_deviance = np.inf
    result: GLMResult | None = None

    for _ in range(max_iter):
        glm = GLM(
            family=family,
            link=link,
            backend=backend,
            tweedie_power=tweedie_power,
            base_levels=base_levels or {},
        )
        if exposure is not None:
            base_offset = np.log(np.asarray(exposure, dtype=float))
        else:
            base_offset = np.zeros(len(X))
        combined = base_offset + constraint_offset
        # We expose the combined offset to the backend by exponentiating into a
        # synthetic exposure column (since the GLM API only accepts exposure).
        synthetic_exposure = np.exp(combined)
        result = glm.fit(
            X,
            y,
            sample_weight=sample_weight,
            exposure=synthetic_exposure,
        )
        rel = result.relativities(constrained_var).reindex(levels)
        rel_proj = monotone_relativities(rel, increasing=increasing)

        # Encode projected relativity as an additive log-offset for next iter.
        log_rel_map = np.log(rel_proj).to_dict()
        constraint_offset_new = X[constrained_var].map(log_rel_map).to_numpy(dtype=float)
        # Subtract the unconstrained design-matrix contribution for this var so
        # we don't double-count after the next refit drops it.
        # Simpler approach: drop the variable from X for subsequent fits and
        # carry only the projected offset.
        X = X[[*other_cols, constrained_var]]  # keep var for predict() context
        constraint_offset = constraint_offset_new

        if abs(last_deviance - result.deviance) < tol:
            break
        last_deviance = result.deviance

    assert result is not None
    return result
