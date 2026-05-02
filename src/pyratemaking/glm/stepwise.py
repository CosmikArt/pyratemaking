"""Stepwise variable selection for GLMs (AIC / BIC).

Forward, backward, and bidirectional flavours. Score = ``deviance + k * p``
with ``k = 2`` for AIC and ``k = ln(n)`` for BIC.

Stepwise selection has well-known shortcomings (selection bias, ignored
correlation between candidates) but remains the standard exam tool, so it
is included here. For production use, prefer penalised regression with
cross-validation (:mod:`pyratemaking.glm.penalized`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyratemaking.glm.backend import GLM, GLMResult


@dataclass
class StepwiseResult:
    """Outcome of a stepwise selection run."""

    selected: list[str]
    history: pd.DataFrame
    final: GLMResult


def _score(result: GLMResult, criterion: str) -> float:
    n = result.n_obs
    p = result.n_features
    if criterion == "aic":
        return result.deviance + 2.0 * p
    if criterion == "bic":
        return result.deviance + np.log(n) * p
    raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion!r}")


def stepwise_select(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    candidates: Sequence[str],
    *,
    direction: str = "forward",
    criterion: str = "aic",
    family: str = "tweedie",
    link: str = "log",
    backend: str = "glum",
    tweedie_power: float = 1.5,
    sample_weight: np.ndarray | pd.Series | None = None,
    exposure: np.ndarray | pd.Series | None = None,
    base_levels: dict[str, str] | None = None,
) -> StepwiseResult:
    """Pick a subset of ``candidates`` that minimises AIC or BIC.

    Parameters
    ----------
    direction : {"forward", "backward", "both"}
        Forward: start empty, add one at a time. Backward: start full, drop
        one at a time. Both: at each step consider every add and every drop.
    """
    candidates = list(candidates)
    chosen: list[str] = [] if direction == "forward" else list(candidates)
    history: list[dict[str, object]] = []

    def fit(cols: list[str]) -> GLMResult:
        if not cols:
            cols = []
        glm = GLM(
            family=family,
            link=link,
            backend=backend,
            tweedie_power=tweedie_power,
            base_levels=base_levels or {},
        )
        x = X[cols] if cols else pd.DataFrame(index=X.index)
        return glm.fit(
            x,
            y,
            sample_weight=sample_weight,
            exposure=exposure,
        )

    current = fit(chosen)
    current_score = _score(current, criterion)
    history.append({"step": 0, "action": "init", "vars": tuple(chosen), criterion: current_score})

    step = 0
    while True:
        step += 1
        candidate_moves: list[tuple[float, str, str, list[str]]] = []
        if direction in ("forward", "both"):
            for v in candidates:
                if v in chosen:
                    continue
                trial = [*chosen, v]
                res = fit(trial)
                candidate_moves.append((_score(res, criterion), "add", v, trial))
        if direction in ("backward", "both"):
            for v in chosen:
                trial = [c for c in chosen if c != v]
                res = fit(trial)
                candidate_moves.append((_score(res, criterion), "drop", v, trial))

        if not candidate_moves:
            break
        candidate_moves.sort(key=lambda t: t[0])
        best_score, action, var, trial = candidate_moves[0]
        if best_score >= current_score - 1e-9:
            break
        chosen = trial
        current = fit(chosen)
        current_score = _score(current, criterion)
        history.append(
            {
                "step": step,
                "action": f"{action}({var})",
                "vars": tuple(chosen),
                criterion: current_score,
            }
        )

    return StepwiseResult(
        selected=chosen,
        history=pd.DataFrame(history),
        final=current,
    )
