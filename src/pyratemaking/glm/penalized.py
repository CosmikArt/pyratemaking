"""Penalised GLMs: ridge, lasso, elastic net.

Backed by ``glum`` because statsmodels does not penalise GLMs natively.
``alpha="cv"`` runs k-fold cross-validation across a log-spaced alpha grid
and picks the value that minimises out-of-sample deviance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pyratemaking.glm.backend import GLM, GLMResult, _deviance, build_design
from pyratemaking.glm.families import family_spec


_PENALTY_RATIO = {
    "ridge": 0.0,
    "l2": 0.0,
    "lasso": 1.0,
    "l1": 1.0,
    "elastic_net": 0.5,
}


def fit_penalized(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    family: str = "tweedie",
    link: str = "log",
    tweedie_power: float = 1.5,
    penalty: str = "ridge",
    alpha: float | str = 0.01,
    l1_ratio: float | None = None,
    sample_weight: np.ndarray | pd.Series | None = None,
    exposure: np.ndarray | pd.Series | None = None,
    base_levels: dict[str, str] | None = None,
    cv_alphas: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    cv_folds: int = 5,
    random_state: int = 0,
) -> GLMResult:
    """Fit a penalised GLM. ``alpha="cv"`` selects alpha by k-fold CV.

    Parameters
    ----------
    penalty : {"ridge", "lasso", "elastic_net"}
        Sets ``l1_ratio`` to 0, 1, or 0.5 unless ``l1_ratio`` is given explicitly.
    alpha : float or "cv"
        Penalty strength.
    cv_alphas : sequence of float
        Grid for cross-validation when ``alpha == "cv"``.
    """
    if l1_ratio is None:
        if penalty not in _PENALTY_RATIO:
            raise ValueError(f"unknown penalty {penalty!r}")
        l1_ratio = _PENALTY_RATIO[penalty]

    if alpha == "cv":
        alpha = _select_alpha_by_cv(
            X,
            y,
            family=family,
            link=link,
            tweedie_power=tweedie_power,
            l1_ratio=l1_ratio,
            sample_weight=sample_weight,
            exposure=exposure,
            base_levels=base_levels,
            alphas=cv_alphas,
            n_folds=cv_folds,
            random_state=random_state,
        )

    glm = GLM(
        family=family,
        link=link,
        backend="glum",
        tweedie_power=tweedie_power,
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        base_levels=base_levels or {},
    )
    return glm.fit(X, y, sample_weight=sample_weight, exposure=exposure)


def _select_alpha_by_cv(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    family: str,
    link: str,
    tweedie_power: float,
    l1_ratio: float,
    sample_weight: np.ndarray | pd.Series | None,
    exposure: np.ndarray | pd.Series | None,
    base_levels: dict[str, str] | None,
    alphas: Sequence[float],
    n_folds: int,
    random_state: int,
) -> float:
    spec = family_spec(family, link, tweedie_power=tweedie_power)
    y_arr = np.asarray(y, dtype=float)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    exp_arr = None if exposure is None else np.asarray(exposure, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores: list[float] = []
    for a in alphas:
        fold_dev = []
        for train_idx, test_idx in kf.split(X):
            glm = GLM(
                family=family,
                link=link,
                backend="glum",
                tweedie_power=tweedie_power,
                alpha=float(a),
                l1_ratio=float(l1_ratio),
                base_levels=base_levels or {},
            )
            x_train = X.iloc[train_idx].reset_index(drop=True)
            x_test = X.iloc[test_idx].reset_index(drop=True)
            sw_train = None if sw is None else sw[train_idx]
            sw_test = None if sw is None else sw[test_idx]
            exp_train = None if exp_arr is None else exp_arr[train_idx]
            exp_test = None if exp_arr is None else exp_arr[test_idx]
            res = glm.fit(
                x_train,
                y_arr[train_idx],
                sample_weight=sw_train,
                exposure=exp_train,
            )
            mu = res.predict(x_test, exposure=exp_test)
            fold_dev.append(_deviance(spec, y_arr[test_idx], mu, sw_test))
        scores.append(float(np.mean(fold_dev)))
    return float(alphas[int(np.argmin(scores))])


def alpha_path(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    family: str = "tweedie",
    link: str = "log",
    tweedie_power: float = 1.5,
    l1_ratio: float = 0.5,
    alphas: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    sample_weight: np.ndarray | pd.Series | None = None,
    exposure: np.ndarray | pd.Series | None = None,
    base_levels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Refit at each alpha and return a coefficient-path DataFrame.

    Useful for diagnostic plots — read off how each coefficient enters or
    exits as the penalty relaxes.
    """
    rows: list[pd.Series] = []
    for a in alphas:
        glm = GLM(
            family=family,
            link=link,
            backend="glum",
            tweedie_power=tweedie_power,
            alpha=float(a),
            l1_ratio=float(l1_ratio),
            base_levels=base_levels or {},
        )
        res = glm.fit(X, y, sample_weight=sample_weight, exposure=exposure)
        rows.append(res.coef_.rename(a))
    out = pd.concat(rows, axis=1)
    out.columns.name = "alpha"
    return out


# expose the design-matrix builder for downstream callers
__all__ = ["alpha_path", "build_design", "fit_penalized"]
