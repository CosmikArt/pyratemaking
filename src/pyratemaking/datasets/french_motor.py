"""French Motor Third-Party Liability dataset (Charpentier 2014).

Source: ``freMTPL2freq`` and ``freMTPL2sev`` from ``CASdatasets`` (R), released
under CC-BY-NC. The first call downloads both files into the local cache;
later calls read straight from disk.

Schema returned (matched by :mod:`pyratemaking.datasets.synthetic`):

* ``policies``: ``policy_id``, ``policy_ay``, ``exposure``, ``earned_premium``
  (set to NaN — the source does not include premium), plus rating variables
  ``area``, ``veh_power``, ``veh_age``, ``driver_age``, ``bonus_malus``,
  ``veh_brand``, ``veh_gas``, ``density``, ``region``.
* ``claims``: ``policy_id``, ``policy_ay``, ``claim_amount``.

References
----------
Charpentier, A. (2014). *Computational Actuarial Science with R*. CRC.
CASdatasets package: ``http://cas.uqam.ca`` (CC-BY-NC).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pyratemaking.datasets._cache import cache_dir


_FREQ_URLS = (
    "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/freMTPL2freq.csv",
    "https://www.openml.org/data/get_csv/22044756/dataset",
)
_SEV_URLS = (
    "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/freMTPL2sev.csv",
)


def _download(urls: tuple[str, ...], target: Path) -> None:
    import urllib.request

    last_error: Exception | None = None
    for url in urls:
        try:
            urllib.request.urlretrieve(url, target)  # noqa: S310 - public dataset
            return
        except Exception as e:  # pragma: no cover - network behaviour
            last_error = e
            continue
    raise OSError(
        f"failed to download {target.name}; last error: {last_error}. "
        f"Manual download: place CSV at {target}"
    )


def _read_freq(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {
        "IDpol": "policy_id",
        "ClaimNb": "claim_count",
        "Exposure": "exposure",
        "Area": "area",
        "VehPower": "veh_power",
        "VehAge": "veh_age",
        "DrivAge": "driver_age",
        "BonusMalus": "bonus_malus",
        "VehBrand": "veh_brand",
        "VehGas": "veh_gas",
        "Density": "density",
        "Region": "region",
    }
    df = df.rename(columns=rename)
    df["policy_ay"] = 2009  # CASdatasets convention — single observation period
    df["earned_premium"] = np.nan
    return df


def _read_sev(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"IDpol": "policy_id", "ClaimAmount": "claim_amount"})
    df["policy_ay"] = 2009
    return df


def load(*, force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (or fetch) the French Motor TPL dataset.

    Parameters
    ----------
    force_download : bool, default False
        Bypass the cache and re-fetch from source.

    Returns
    -------
    (policies, claims) : tuple of DataFrames

    Raises
    ------
    OSError
        Network failure with no usable cache.
    """
    cache = cache_dir()
    freq_path = cache / "freMTPL2freq.csv"
    sev_path = cache / "freMTPL2sev.csv"

    if force_download or not freq_path.exists():
        _download(_FREQ_URLS, freq_path)
    if force_download or not sev_path.exists():
        _download(_SEV_URLS, sev_path)

    policies = _read_freq(freq_path)
    claims = _read_sev(sev_path)
    return policies, claims


def schema() -> dict[str, list[str]]:
    """Column descriptions for both frames."""
    return {
        "policies": [
            "policy_id",
            "policy_ay",
            "exposure",
            "earned_premium",
            "area",
            "veh_power",
            "veh_age",
            "driver_age",
            "bonus_malus",
            "veh_brand",
            "veh_gas",
            "density",
            "region",
            "claim_count",
        ],
        "claims": ["policy_id", "policy_ay", "claim_amount"],
    }
