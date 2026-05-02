"""Layer pricing using a parametric severity distribution (optional bridge).

When the optional ``actudist`` package is available, severity layers can be
priced analytically — closed-form for lognormal / Pareto, numerical for
mixtures. Without it, fall back to Monte-Carlo on the empirical loss
distribution via :func:`layer_loss_cost`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pyratemaking.large_loss.increased_limits import layer_loss_cost


def layer_pricing_from_distribution(
    *,
    distribution: Any,
    attachment: float,
    limit: float | None,
    n_samples: int = 100_000,
    random_state: int = 0,
) -> dict[str, float]:
    """Price a layer from a severity distribution.

    Parameters
    ----------
    distribution : object
        Either an :mod:`actudist` distribution exposing ``rvs(size, random_state)``,
        or a frozen :mod:`scipy.stats` distribution. The duck-typing check
        is intentional so callers can pass either.
    attachment : float
    limit : float, optional
        Layer width. ``None`` for an unlimited excess layer.

    Returns
    -------
    dict with ``layer_loss_cost`` and ``frequency`` (probability of any
    loss penetrating the layer).
    """
    if hasattr(distribution, "rvs"):
        try:
            sample = distribution.rvs(size=n_samples, random_state=random_state)
        except TypeError:
            rng = np.random.default_rng(random_state)
            sample = distribution.rvs(size=n_samples, random_state=rng)
    else:
        raise TypeError(
            "distribution must expose .rvs(size, random_state); "
            "consider scipy.stats or actudist distributions"
        )
    sample = np.asarray(sample, dtype=float)
    cost = layer_loss_cost(sample, attachment=attachment, limit=limit)
    freq = float((sample > attachment).mean())
    return {"layer_loss_cost": cost, "frequency": freq}


def layer_pricing_table(
    losses: np.ndarray | pd.Series,
    *,
    layers: list[tuple[float, float | None]],
) -> pd.DataFrame:
    """Empirical layer pricing across multiple layers."""
    arr = np.asarray(losses, dtype=float)
    rows = []
    for attachment, width in layers:
        cost = layer_loss_cost(arr, attachment=attachment, limit=width)
        rows.append(
            {
                "attachment": attachment,
                "limit": width,
                "avg_layer_loss": cost,
                "freq_in_layer": float((arr > attachment).mean()),
            }
        )
    return pd.DataFrame(rows)
