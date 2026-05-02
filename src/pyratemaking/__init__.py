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


_LAZY = {"RatePlan": ("pyratemaking.core.plan", "RatePlan")}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        module_name, attr = _LAZY[name]
        return getattr(import_module(module_name), attr)
    raise AttributeError(f"module 'pyratemaking' has no attribute {name!r}")
