"""Bundled and downloadable ratemaking datasets.

* :func:`french_motor.load` — French Motor TPL (Charpentier 2014, CASdatasets,
  CC-BY-NC). Fetched from a public mirror on first call, cached locally.
* :func:`synthetic.generate` — parametrised generator that produces
  realistic policies and claims with the same schema as ``french_motor``,
  so swapping is a one-line change.

Cache lives at ``~/.pyratemaking/data/`` and can be cleared with
:func:`_cache.clear`.
"""

from pyratemaking.datasets import french_motor, synthetic

__all__ = ["french_motor", "synthetic"]
