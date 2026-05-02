"""Unified GLM API over ``glum`` and ``statsmodels``.

Both engines accept the same arguments here; the design matrix is built
once and reused. Categorical columns become drop-first one-hot blocks;
numeric columns pass through untouched. The intercept is always included.

Standard usage:

>>> glm = GLM(family="poisson", link="log")
>>> result = glm.fit(X=policies[["driver_age", "region"]], y=counts,
...                  exposure=policies["earned_exposure"])
>>> result.coef_
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pyratemaking.glm.families import FamilyLink, family_spec


@dataclass
class DesignMatrix:
    """Pandas-aware design matrix with column names preserved."""

    matrix: pd.DataFrame
    base_levels: dict[str, str]
    categorical: list[str]
    numeric: list[str]

    @property
    def feature_names(self) -> list[str]:
        return list(self.matrix.columns)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the same encoding to a new frame, preserving column order."""
        out = build_design(
            X,
            categorical=self.categorical,
            numeric=self.numeric,
            base_levels=self.base_levels,
        )
        missing = [c for c in self.feature_names if c not in out.matrix.columns]
        for c in missing:
            out.matrix[c] = 0.0
        return out.matrix.reindex(columns=self.feature_names, fill_value=0.0)


def _looks_categorical(s: pd.Series) -> bool:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if s.dtype == object:
        return True
    return bool(pd.api.types.is_string_dtype(s) and not pd.api.types.is_numeric_dtype(s))


def build_design(
    X: pd.DataFrame,
    *,
    categorical: list[str] | None = None,
    numeric: list[str] | None = None,
    base_levels: dict[str, str] | None = None,
    add_intercept: bool = True,
) -> DesignMatrix:
    """Build a design matrix with drop-first one-hot for categoricals.

    Categorical columns are detected automatically (object / category dtype)
    when ``categorical`` is omitted; everything else is treated as numeric.
    """
    if categorical is None and numeric is None:
        categorical = [c for c in X.columns if _looks_categorical(X[c])]
        numeric = [c for c in X.columns if c not in categorical]
    categorical = list(categorical or [])
    numeric = list(numeric or [])

    base_levels = dict(base_levels or {})
    pieces: list[pd.DataFrame] = []
    if add_intercept:
        pieces.append(pd.DataFrame({"Intercept": np.ones(len(X))}, index=X.index))

    for c in categorical:
        col = X[c].astype("category")
        levels = list(col.cat.categories)
        base = base_levels.get(c, levels[0])
        if base not in levels:
            raise ValueError(f"base level {base!r} not in column {c!r} categories")
        base_levels[c] = base
        non_base = [lvl for lvl in levels if lvl != base]
        for lvl in non_base:
            pieces.append(
                pd.DataFrame(
                    {f"{c}[{lvl}]": (col == lvl).astype(float).to_numpy()},
                    index=X.index,
                )
            )

    for c in numeric:
        pieces.append(pd.DataFrame({c: X[c].astype(float).to_numpy()}, index=X.index))

    matrix = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=X.index)
    return DesignMatrix(
        matrix=matrix,
        base_levels=base_levels,
        categorical=categorical,
        numeric=numeric,
    )


@dataclass
class GLMResult:
    """Fitted GLM result with named coefficients."""

    coef_: pd.Series
    family: str
    link: str
    backend: str
    deviance: float
    n_obs: int
    n_features: int
    design: DesignMatrix
    raw_estimator: object = field(default=None, repr=False)
    fitted_offset: bool = False
    tweedie_power: float | None = None

    @property
    def intercept_(self) -> float:
        return float(self.coef_.get("Intercept", 0.0))

    def predict(
        self,
        X: pd.DataFrame,
        *,
        exposure: np.ndarray | pd.Series | None = None,
    ) -> np.ndarray:
        """Return mean predictions on the response scale."""
        design = self.design.transform(X).to_numpy(dtype=float)
        coefs = self.coef_.reindex(self.design.feature_names).to_numpy(dtype=float)
        eta = design @ coefs
        if self.fitted_offset and exposure is None:
            raise ValueError(
                "this fit used an exposure offset; pass exposure= to predict"
            )
        if exposure is not None:
            eta = eta + np.log(np.asarray(exposure, dtype=float))
        return _inverse_link(eta, self.link)

    def relativities(self, variable: str) -> pd.Series:
        """Multiplicative relativities for a single categorical variable."""
        if self.link != "log":
            raise ValueError("relativities are only defined for log-link models")
        prefix = f"{variable}["
        items: dict[str, float] = {self.design.base_levels[variable]: 1.0}
        for name, coef in self.coef_.items():
            if isinstance(name, str) and name.startswith(prefix):
                level = name[len(prefix):-1]
                items[level] = float(np.exp(coef))
        return pd.Series(items, name=f"relativity[{variable}]")

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.coef_,
                "exp_coef": np.exp(self.coef_) if self.link == "log" else np.nan,
            }
        )

    def __repr__(self) -> str:
        return (
            f"GLMResult(family={self.family!r}, link={self.link!r}, "
            f"backend={self.backend!r}, n_obs={self.n_obs}, n_features={self.n_features})"
        )


@dataclass
class GLM:
    """Configurable GLM fitter that picks a backend at fit time."""

    family: str = "tweedie"
    link: str = "log"
    backend: str = "glum"
    tweedie_power: float = 1.5
    fit_intercept: bool = True
    alpha: float | str | None = None
    l1_ratio: float = 0.0
    base_levels: dict[str, str] = field(default_factory=dict)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        sample_weight: np.ndarray | pd.Series | None = None,
        exposure: np.ndarray | pd.Series | None = None,
        categorical: list[str] | None = None,
        numeric: list[str] | None = None,
    ) -> GLMResult:
        """Fit and return a :class:`GLMResult`."""
        spec = family_spec(self.family, self.link, tweedie_power=self.tweedie_power)
        design = build_design(
            X,
            categorical=categorical,
            numeric=numeric,
            base_levels=self.base_levels,
            add_intercept=self.fit_intercept,
        )
        y_arr = np.asarray(y, dtype=float)
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        offset = (
            np.log(np.asarray(exposure, dtype=float))
            if exposure is not None and self.link == "log"
            else None
        )
        if self.backend == "glum":
            return self._fit_glum(spec, design, y_arr, sw, offset)
        if self.backend == "statsmodels":
            return self._fit_statsmodels(spec, design, y_arr, sw, offset)
        raise ValueError(
            f"backend must be 'glum' or 'statsmodels', got {self.backend!r}"
        )

    def _fit_glum(
        self,
        spec: FamilyLink,
        design: DesignMatrix,
        y: np.ndarray,
        sw: np.ndarray | None,
        offset: np.ndarray | None,
    ) -> GLMResult:
        from glum import GeneralizedLinearRegressor, TweedieDistribution

        family_arg: object
        if spec.name == "tweedie":
            family_arg = TweedieDistribution(power=spec.tweedie_power)
        elif spec.name == "inverse_gaussian":
            family_arg = "inverse.gaussian"
        else:
            family_arg = spec.name

        alpha = 0.0 if self.alpha in (None, "cv") else float(self.alpha)  # type: ignore[arg-type]
        # glum requires fit_intercept=False when an Intercept column is in X.
        glm = GeneralizedLinearRegressor(
            family=family_arg,
            link=spec.link,
            fit_intercept=False,
            alpha=alpha,
            l1_ratio=self.l1_ratio,
        )
        X_arr = design.matrix.to_numpy(dtype=float)
        glm.fit(X_arr, y, sample_weight=sw, offset=offset)
        coefs = pd.Series(glm.coef_, index=design.feature_names, name="coef")
        deviance = _deviance(spec, y, glm.predict(X_arr, offset=offset), sw)
        return GLMResult(
            coef_=coefs,
            family=spec.name,
            link=spec.link,
            backend="glum",
            deviance=deviance,
            n_obs=int(y.shape[0]),
            n_features=design.matrix.shape[1],
            design=design,
            raw_estimator=glm,
            fitted_offset=offset is not None,
            tweedie_power=spec.tweedie_power,
        )

    def _fit_statsmodels(
        self,
        spec: FamilyLink,
        design: DesignMatrix,
        y: np.ndarray,
        sw: np.ndarray | None,
        offset: np.ndarray | None,
    ) -> GLMResult:
        import statsmodels.api as sm

        sm_family = spec.statsmodels_factory()
        glm = sm.GLM(
            y,
            design.matrix.to_numpy(dtype=float),
            family=sm_family,
            offset=offset,
            freq_weights=sw,
        )
        res = glm.fit()
        coefs = pd.Series(res.params, index=design.feature_names, name="coef")
        return GLMResult(
            coef_=coefs,
            family=spec.name,
            link=spec.link,
            backend="statsmodels",
            deviance=float(res.deviance),
            n_obs=int(y.shape[0]),
            n_features=design.matrix.shape[1],
            design=design,
            raw_estimator=res,
            fitted_offset=offset is not None,
            tweedie_power=spec.tweedie_power,
        )


def _inverse_link(eta: np.ndarray, link: str) -> np.ndarray:
    if link == "log":
        return np.exp(eta)
    if link == "identity":
        return eta
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-eta))
    if link == "inverse":
        return 1.0 / eta
    if link == "inverse_squared":
        return 1.0 / np.sqrt(eta)
    raise ValueError(f"unknown link {link!r}")


def _deviance(
    spec: FamilyLink,
    y: np.ndarray,
    mu: np.ndarray,
    sw: np.ndarray | None,
) -> float:
    """Family-specific deviance, used for AIC / BIC and reporting."""
    w = np.ones_like(y) if sw is None else sw
    eps = 1e-12
    if spec.name == "poisson":
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(y / np.where(mu > 0, mu, eps)), 0.0)
        dev = 2.0 * np.sum(w * (term - (y - mu)))
    elif spec.name == "gamma":
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, np.log(y / np.where(mu > 0, mu, eps)), 0.0)
        dev = 2.0 * np.sum(w * (-term + (y - mu) / np.where(mu > 0, mu, eps)))
    elif spec.name == "tweedie":
        p = spec.tweedie_power or 1.5
        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = np.where(
                y > 0, y ** (2 - p) / ((1 - p) * (2 - p)), 0.0
            ) - y * mu ** (1 - p) / (1 - p)
            term2 = mu ** (2 - p) / (2 - p)
        dev = 2.0 * np.sum(w * (term1 + term2))
    elif spec.name == "gaussian":
        dev = float(np.sum(w * (y - mu) ** 2))
    else:
        dev = float(np.sum(w * (y - mu) ** 2))
    return float(dev)
