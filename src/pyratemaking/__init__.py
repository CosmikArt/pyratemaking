"""Ratemaking workflow for P&C insurance.

The package follows the methodology in Werner & Modlin (2016), *Basic Ratemaking*
(CAS, 5th ed.). The public entry point is :class:`pyratemaking.RatePlan`, which
composes loss-ratio indication, on-leveling, trending, development, classification
GLMs, large-loss procedures, and rate implementation into one reproducible pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyratemaking._version import __version__

if TYPE_CHECKING:
    from pyratemaking.core.plan import RatePlan

__all__ = ["RatePlan", "__version__"]


def __getattr__(name: str) -> Any:
    if name == "RatePlan":
        from pyratemaking.core.plan import RatePlan as _RP

        return _RP
    raise AttributeError(f"module 'pyratemaking' has no attribute {name!r}")
