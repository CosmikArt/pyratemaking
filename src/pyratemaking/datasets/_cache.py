"""Cache directory helpers for downloaded datasets."""

from __future__ import annotations

from pathlib import Path
import shutil


def cache_dir() -> Path:
    """Return the local cache root, creating it if needed."""
    p = Path.home() / ".pyratemaking" / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def clear() -> None:
    """Remove the cache directory."""
    p = Path.home() / ".pyratemaking" / "data"
    if p.exists():
        shutil.rmtree(p)
