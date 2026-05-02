"""Chain ladder, Bornhuetter-Ferguson, and Cape Cod methods.

The triangle is a DataFrame indexed by accident year, columns are integer
development ages (months or years), values are cumulative paid or reported
losses. Cells beyond the diagonal are NaN.

References
----------
Mack, T. (1993). "Distribution-free Calculation of the Standard Error of
Chain Ladder Reserve Estimates." *ASTIN Bulletin*, 23(2), 213–225.

Friedland, J. (2013). *Fundamentals of General Insurance Actuarial Analysis*,
§§ 11–13.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd


def age_to_age_factors(
    triangle: pd.DataFrame,
    *,
    weighted: bool = True,
) -> pd.Series:
    """Age-to-age (link) factors for a cumulative loss triangle.

    Parameters
    ----------
    triangle : DataFrame
        Cumulative losses, AY × dev-age, with NaN beyond the diagonal.
    weighted : bool, default True
        Volume-weighted factors (Mack 1993): ``f_j = sum(C[:, j+1]) / sum(C[:, j])``
        across rows where both ``j`` and ``j+1`` are observed. When False,
        returns the simple average of individual ratios.

    Returns
    -------
    Series indexed by ``f"{j}-{j+1}"`` with the link factors.
    """
    cols = list(triangle.columns)
    out = {}
    for j, j_next in itertools.pairwise(cols):
        prev = triangle[j]
        nxt = triangle[j_next]
        mask = prev.notna() & nxt.notna()
        if not mask.any():
            out[f"{j}-{j_next}"] = np.nan
            continue
        if weighted:
            num = float(nxt[mask].sum())
            den = float(prev[mask].sum())
            out[f"{j}-{j_next}"] = num / den if den != 0 else np.nan
        else:
            ratios = nxt[mask] / prev[mask]
            out[f"{j}-{j_next}"] = float(ratios.mean())
    return pd.Series(out, name="link_factor")


def cumulative_factors(
    link_factors: pd.Series,
    *,
    tail: float = 1.0,
) -> pd.Series:
    """Convert link factors to age-to-ultimate cumulative factors.

    The last entry includes the tail. Index is the *development age* at
    which the factor brings cumulative losses to ultimate.
    """
    arr = np.asarray(link_factors.to_numpy(dtype=float), dtype=float)
    cdfs = np.empty(arr.size + 1)
    cdfs[-1] = float(tail)
    for k in range(arr.size - 1, -1, -1):
        cdfs[k] = arr[k] * cdfs[k + 1]

    dev_ages = []
    for label in link_factors.index:
        left, _ = label.split("-")
        dev_ages.append(_int_or_str(left))
    last_label = link_factors.index[-1]
    _, right = last_label.split("-")
    dev_ages.append(_int_or_str(right))

    return pd.Series(cdfs, index=pd.Index(dev_ages, name="dev_age"), name="cdf_to_ult")


def _int_or_str(label: str) -> int | str:
    try:
        return int(label)
    except ValueError:
        try:
            return int(float(label))
        except ValueError:
            return label


@dataclass
class ChainLadder:
    """Chain ladder ultimate-loss projection.

    Parameters
    ----------
    triangle : DataFrame
        Cumulative losses, AY × development age, with NaN beyond the diagonal.
    tail_factor : float, default 1.0
        Multiplicative tail factor applied to the last observed cumulative loss.
    weighted : bool, default True
        Use volume-weighted link factors (Mack 1993). If False, simple averages.
    """

    triangle: pd.DataFrame
    tail_factor: float = 1.0
    weighted: bool = True

    def __post_init__(self) -> None:
        self._link = age_to_age_factors(self.triangle, weighted=self.weighted)
        self._cdf = cumulative_factors(self._link, tail=self.tail_factor)

    @property
    def link_factors(self) -> pd.Series:
        return self._link

    @property
    def cdf(self) -> pd.Series:
        return self._cdf

    def ultimates(self) -> pd.DataFrame:
        """Project each AY to ultimate using its latest observed diagonal cell."""
        rows = []
        for ay, row in self.triangle.iterrows():
            obs = row.dropna()
            if obs.empty:
                continue
            last_age = obs.index[-1]
            last_value = float(obs.iloc[-1])
            cdf = float(self._cdf.loc[last_age])
            ultimate = last_value * cdf
            rows.append(
                {
                    "ay": ay,
                    "latest_age": last_age,
                    "latest_cumulative": last_value,
                    "cdf_to_ult": cdf,
                    "ultimate": ultimate,
                    "reserve": ultimate - last_value,
                }
            )
        return pd.DataFrame(rows).set_index("ay")

    def __repr__(self) -> str:
        return f"ChainLadder(n_periods={len(self.triangle)}, tail_factor={self.tail_factor:.4f})"

    def to_burncost(self):  # pragma: no cover - depends on optional package
        from pyratemaking.development._burncost_bridge import to_burncost_triangle

        return to_burncost_triangle(self.triangle)


@dataclass
class BornhuetterFerguson:
    """Bornhuetter-Ferguson ultimate-loss projection.

    Blends an a priori ultimate (typically loss-ratio × premium) with
    reported development. The reported portion is taken from the data, the
    unreported portion from the a priori.
    """

    triangle: pd.DataFrame
    a_priori_ultimate: pd.Series
    tail_factor: float = 1.0
    weighted: bool = True

    def __post_init__(self) -> None:
        self._link = age_to_age_factors(self.triangle, weighted=self.weighted)
        self._cdf = cumulative_factors(self._link, tail=self.tail_factor)

    @property
    def cdf(self) -> pd.Series:
        return self._cdf

    def ultimates(self) -> pd.DataFrame:
        rows = []
        for ay, row in self.triangle.iterrows():
            obs = row.dropna()
            if obs.empty or ay not in self.a_priori_ultimate.index:
                continue
            last_age = obs.index[-1]
            last_value = float(obs.iloc[-1])
            cdf = float(self._cdf.loc[last_age])
            pct_reported = 1.0 / cdf if cdf != 0 else 0.0
            a_priori = float(self.a_priori_ultimate.loc[ay])
            expected_unreported = a_priori * (1.0 - pct_reported)
            ultimate = last_value + expected_unreported
            rows.append(
                {
                    "ay": ay,
                    "latest_age": last_age,
                    "latest_cumulative": last_value,
                    "a_priori_ultimate": a_priori,
                    "pct_reported": pct_reported,
                    "expected_unreported": expected_unreported,
                    "ultimate": ultimate,
                    "reserve": ultimate - last_value,
                }
            )
        return pd.DataFrame(rows).set_index("ay")


@dataclass
class CapeCod:
    """Cape Cod ultimate-loss projection.

    Estimates the expected loss ratio from the data itself, then plugs it
    into the BF formula.

    Parameters
    ----------
    triangle : DataFrame
    used_premium : Series
        On-level earned premium by AY.
    tail_factor : float, default 1.0
    weighted : bool, default True
    decay : float, default 0.0
        Friedland §13.5 — when ``> 0``, AYs further from the latest diagonal
        receive less weight in the ELR estimate, reflecting trend uncertainty.
    """

    triangle: pd.DataFrame
    used_premium: pd.Series
    tail_factor: float = 1.0
    weighted: bool = True
    decay: float = 0.0

    def __post_init__(self) -> None:
        self._link = age_to_age_factors(self.triangle, weighted=self.weighted)
        self._cdf = cumulative_factors(self._link, tail=self.tail_factor)

    @property
    def cdf(self) -> pd.Series:
        return self._cdf

    @property
    def expected_loss_ratio(self) -> float:
        rows_for_elr = self._reported_per_ay()
        latest_ay = max(rows_for_elr.keys()) if rows_for_elr else 0
        num = 0.0
        den = 0.0
        for ay, (reported, pct_reported) in rows_for_elr.items():
            if ay not in self.used_premium.index:
                continue
            premium = float(self.used_premium.loc[ay])
            weight = (1 - self.decay) ** (latest_ay - ay) if self.decay > 0 else 1.0
            num += weight * reported
            den += weight * premium * pct_reported
        return num / den if den != 0 else 0.0

    def _reported_per_ay(self) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for ay, row in self.triangle.iterrows():
            obs = row.dropna()
            if obs.empty:
                continue
            last_age = obs.index[-1]
            last_value = float(obs.iloc[-1])
            cdf = float(self._cdf.loc[last_age])
            out[int(ay)] = (last_value, 1.0 / cdf if cdf != 0 else 0.0)
        return out

    def ultimates(self) -> pd.DataFrame:
        elr = self.expected_loss_ratio
        rows = []
        for ay, (reported, pct_reported) in self._reported_per_ay().items():
            if ay not in self.used_premium.index:
                continue
            premium = float(self.used_premium.loc[ay])
            expected_unreported = premium * elr * (1 - pct_reported)
            ultimate = reported + expected_unreported
            rows.append(
                {
                    "ay": ay,
                    "used_premium": premium,
                    "elr": elr,
                    "latest_cumulative": reported,
                    "pct_reported": pct_reported,
                    "expected_unreported": expected_unreported,
                    "ultimate": ultimate,
                    "reserve": ultimate - reported,
                }
            )
        return pd.DataFrame(rows).set_index("ay")
