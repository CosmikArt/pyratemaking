"""GAM wrapper (optional).

A thin shim around ``pygam`` so non-linear effects on continuous rating
variables (driver age, vehicle age, sum insured) can be fit with the same
ratemaking-friendly defaults: log link, GCV smoothing, weights via
exposure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def fit_gam(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    smooth_columns: Sequence[str],
    family: str = "gamma",
    weights: np.ndarray | pd.Series | None = None,
):
    """Fit a GAM with smooth terms on the requested columns.

    Returns the underlying ``pygam`` model so callers can use ``predict``,
    ``partial_dependence``, etc. natively.
    """
    try:
        from pygam import GammaGAM, LinearGAM, PoissonGAM, TweedieGAM, s
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "GAMs require pygam. Install with: pip install pyratemaking[gam]"
        ) from exc

    spec = None
    cols = list(X.columns)
    for i, c in enumerate(cols):
        term = s(i) if c in smooth_columns else None
        if term is None:
            continue
        spec = term if spec is None else spec + term

    factory = {
        "gamma": GammaGAM,
        "poisson": PoissonGAM,
        "tweedie": TweedieGAM,
        "gaussian": LinearGAM,
    }
    if family not in factory:
        raise ValueError(f"unsupported family for GAM: {family!r}")

    gam = factory[family](spec) if spec is not None else factory[family]()
    weights_arr = None if weights is None else np.asarray(weights, dtype=float)
    gam.fit(X.to_numpy(dtype=float), np.asarray(y, dtype=float), weights=weights_arr)
    return gam
