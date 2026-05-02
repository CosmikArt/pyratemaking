"""Top-level ratemaking workflow objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyratemaking.core.classification import ClassificationResult
    from pyratemaking.core.implementation import ImplementationResult
    from pyratemaking.core.indication import Indication
    from pyratemaking.core.plan import RatePlan

__all__ = [
    "ClassificationResult",
    "ImplementationResult",
    "Indication",
    "RatePlan",
]


_LAZY = {
    "ClassificationResult": ("pyratemaking.core.classification", "ClassificationResult"),
    "ImplementationResult": ("pyratemaking.core.implementation", "ImplementationResult"),
    "Indication": ("pyratemaking.core.indication", "Indication"),
    "RatePlan": ("pyratemaking.core.plan", "RatePlan"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        from importlib import import_module

        return getattr(import_module(module_name), attr)
    raise AttributeError(f"module 'pyratemaking.core' has no attribute {name!r}")
