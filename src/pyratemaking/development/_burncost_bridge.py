"""Optional bridge to the ``burncost`` package.

When present, expose a converter so :class:`ChainLadder` results can flow
into burning-cost analyses without manual reshaping.
"""

from __future__ import annotations

import pandas as pd


def to_burncost_triangle(triangle: pd.DataFrame, *, cumulative: bool = True):
    """Convert a pandas triangle to a :class:`burncost.LossTriangle`.

    Parameters
    ----------
    triangle : DataFrame
        AY × dev-age cumulative loss triangle as produced by
        :class:`pyratemaking.development.ChainLadder` or :func:`io.claims.loss_triangle`.
    cumulative : bool, default True
        Set to False if ``triangle`` holds incremental losses.
    """
    try:
        from burncost import LossTriangle
    except ImportError as exc:
        raise ImportError(
            "burncost is not installed. Install with: pip install pyratemaking[burncost]"
        ) from exc

    return LossTriangle(
        triangle.to_numpy(dtype=float),
        accident_years=[int(x) for x in triangle.index],
        dev_periods=[int(x) for x in triangle.columns],
        cumulative=cumulative,
    )
