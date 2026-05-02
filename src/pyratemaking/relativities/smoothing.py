"""Smooth relativities along an ordered dimension.

Built-in default is a centred moving average. When the optional ``whsmooth``
package is present, callers can pass ``method="whittaker"`` to get
Whittaker-Henderson smoothing weighted by exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_relativities(
    relativities: pd.Series,
    *,
    method: str = "moving_average",
    window: int = 3,
    weights: pd.Series | None = None,
    smoothing_param: float = 100.0,
) -> pd.Series:
    """Smooth a relativity series.

    Parameters
    ----------
    relativities : Series
        Ordered series of relativities (e.g. by driver age band).
    method : {"moving_average", "whittaker"}
        ``"whittaker"`` requires the optional ``whsmooth`` package.
    window : int, default 3
        Centred moving average window. Used only with ``moving_average``.
    weights : Series, optional
        Per-level weights (e.g. exposure). Used only with ``whittaker``.
    smoothing_param : float, default 100
        Whittaker-Henderson smoothing strength. Higher = smoother.
    """
    rel = relativities.astype(float)
    if method == "moving_average":
        return rel.rolling(window=window, center=True, min_periods=1).mean()
    if method == "whittaker":
        return _whittaker(rel, weights=weights, smoothing_param=smoothing_param)
    raise ValueError(f"unknown smoothing method {method!r}")


def _whittaker(  # pragma: no cover - depends on optional package
    rel: pd.Series,
    *,
    weights: pd.Series | None,
    smoothing_param: float,
) -> pd.Series:
    try:
        import whsmooth
    except ImportError as exc:
        raise ImportError(
            "Whittaker smoothing requires whsmooth. "
            "Install with: pip install pyratemaking[smoothing]"
        ) from exc

    if not hasattr(whsmooth, "smooth"):
        raise AttributeError("whsmooth.smooth not found in installed version")

    w = (
        weights.reindex(rel.index).fillna(0).to_numpy(dtype=float)
        if weights is not None
        else np.ones(len(rel))
    )
    smoothed = whsmooth.smooth(
        rel.to_numpy(dtype=float),
        weights=w,
        lam=float(smoothing_param),
    )
    return pd.Series(np.asarray(smoothed, dtype=float), index=rel.index, name=rel.name)
