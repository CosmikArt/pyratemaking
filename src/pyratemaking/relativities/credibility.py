"""Credibility-weight relativities against a complement.

Lightweight built-in: square-root rule (``Z = sqrt(n / N_full)``) for the
default. When the optional ``actuarcredibility`` package is installed, the
caller can swap in empirical Bayes / Buhlmann-Straub credibility via the
``method`` argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def credibility_weighted(
    relativities: pd.Series,
    exposures: pd.Series,
    *,
    complement: float | pd.Series = 1.0,
    full_credibility_exposure: float = 1082.0,
    method: str = "square_root",
) -> pd.Series:
    """Credibility-weight relativities against a complement.

    Parameters
    ----------
    relativities : Series
        Indicated relativities by level.
    exposures : Series
        Earned exposure per level. Index must align with ``relativities``.
    complement : float or Series, default 1.0
        Complement of credibility (e.g. countrywide trend = 1.0).
    full_credibility_exposure : float, default 1082.0
        Standard for full credibility. Default is the classical ``1082`` claims
        for ±5% accuracy at 90% confidence (Longley-Cook).
    method : str, default ``"square_root"``
        ``"square_root"`` for the classical rule. ``"actuarcredibility"`` to
        delegate to that package when installed.
    """
    if not relativities.index.equals(exposures.index):
        relativities, exposures = relativities.align(exposures, join="inner")
    n = exposures.to_numpy(dtype=float)

    if method == "square_root":
        z = np.minimum(np.sqrt(n / max(full_credibility_exposure, 1e-12)), 1.0)
    elif method == "actuarcredibility":
        z = _actuarcredibility_z(relativities, exposures)
    else:
        raise ValueError(f"unknown credibility method {method!r}")

    if isinstance(complement, pd.Series):
        comp = complement.reindex(relativities.index).to_numpy(dtype=float)
    else:
        comp = np.full(len(relativities), float(complement))

    weighted = z * relativities.to_numpy(dtype=float) + (1 - z) * comp
    return pd.Series(weighted, index=relativities.index, name="credibility_weighted")


def _actuarcredibility_z(
    relativities: pd.Series, exposures: pd.Series
) -> np.ndarray:  # pragma: no cover - depends on optional package
    try:
        import actuarcredibility
    except ImportError as exc:
        raise ImportError(
            "actuarcredibility is not installed. "
            "Install with: pip install pyratemaking[credibility]"
        ) from exc
    if not hasattr(actuarcredibility, "buhlmann_z"):
        raise AttributeError(
            "actuarcredibility.buhlmann_z not found in installed version"
        )
    return np.asarray(
        actuarcredibility.buhlmann_z(relativities, exposures), dtype=float
    )
