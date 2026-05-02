"""Optional bridge to the ``burncost`` package.

When present, expose a converter so :class:`ChainLadder` results can flow
into burning-cost analyses without manual reshaping.
"""

from __future__ import annotations

import pandas as pd


def to_burncost_triangle(triangle: pd.DataFrame):
    """Hand a triangle to the optional ``burncost`` package."""
    try:
        import burncost
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "burncost is not installed. Install with: pip install pyratemaking[burncost]"
        ) from exc

    if not hasattr(burncost, "Triangle"):
        raise AttributeError(
            "the installed burncost version does not expose Triangle; check the docs"
        )
    return burncost.Triangle(triangle)
