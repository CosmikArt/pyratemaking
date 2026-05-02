"""Classification relativities (W&M §§9–10).

* :func:`one_way_relativities` — pure-premium relativities at the marginal
  level. Useful as a sanity check before fitting a multi-way GLM.
* :func:`multi_way_relativities` — pulls relativities from a fitted
  :class:`pyratemaking.glm.GLMResult`.
* :func:`credibility_weighted` — soft bridge to ``actuarcredibility`` for
  empirical Bayes credibility on relativities by exposure.
* :func:`smooth_relativities` — soft bridge to ``whsmooth`` for smoothing
  relativities along an ordered dimension (age, deductible, territory).
"""

from pyratemaking.relativities.credibility import credibility_weighted
from pyratemaking.relativities.multi_way import multi_way_relativities
from pyratemaking.relativities.one_way import one_way_relativities
from pyratemaking.relativities.smoothing import smooth_relativities

__all__ = [
    "credibility_weighted",
    "multi_way_relativities",
    "one_way_relativities",
    "smooth_relativities",
]
