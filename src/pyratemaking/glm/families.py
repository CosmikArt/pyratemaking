"""Family specs translated between glum and statsmodels.

Each backend names its families slightly differently. ``family_spec`` returns
both representations so the backend adapter can pick whichever is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FamilyLink:
    """Joint family + link spec, in both engines' native vocabulary."""

    name: str
    link: str
    glum_family: str | Any
    statsmodels_factory: Any
    tweedie_power: float | None = None


def family_spec(
    family: str,
    link: str = "log",
    *,
    tweedie_power: float = 1.5,
) -> FamilyLink:
    """Return a :class:`FamilyLink` for the requested family / link.

    Supported families: ``poisson``, ``gamma``, ``tweedie``, ``inverse_gaussian``,
    ``binomial``, ``gaussian``.
    """
    family = family.lower()
    link = link.lower()

    import statsmodels.api as sm

    links_map = {
        "log": sm.families.links.Log(),
        "identity": sm.families.links.Identity(),
        "logit": sm.families.links.Logit(),
        "inverse": sm.families.links.InversePower(),
        "inverse_squared": sm.families.links.InverseSquared(),
    }
    if link not in links_map:
        raise ValueError(f"unsupported link {link!r}; choose from {sorted(links_map)}")

    if family == "poisson":
        sm_factory = lambda: sm.families.Poisson(link=links_map[link])  # noqa: E731
        return FamilyLink(family, link, "poisson", sm_factory)
    if family == "gamma":
        sm_factory = lambda: sm.families.Gamma(link=links_map[link])  # noqa: E731
        return FamilyLink(family, link, "gamma", sm_factory)
    if family == "tweedie":
        sm_factory = lambda: sm.families.Tweedie(  # noqa: E731
            var_power=tweedie_power, link=links_map[link]
        )
        return FamilyLink(family, link, "tweedie", sm_factory, tweedie_power=tweedie_power)
    if family in ("inverse_gaussian", "ig"):
        sm_factory = lambda: sm.families.InverseGaussian(link=links_map[link])  # noqa: E731
        return FamilyLink("inverse_gaussian", link, "inverse.gaussian", sm_factory)
    if family == "binomial":
        sm_factory = lambda: sm.families.Binomial(link=links_map[link])  # noqa: E731
        return FamilyLink(family, link, "binomial", sm_factory)
    if family == "gaussian":
        sm_factory = lambda: sm.families.Gaussian(link=links_map[link])  # noqa: E731
        return FamilyLink(family, link, "normal", sm_factory)

    raise ValueError(f"unsupported family {family!r}")
