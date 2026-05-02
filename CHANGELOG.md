# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-05-02

First release.

### Added

- `pyratemaking.RatePlan` — end-to-end orchestrator covering indication, classification, implementation, diagnostics, and reporting.
- `pyratemaking.io` — schema validation for policies and claims, AY aggregation, loss-triangle helpers.
- `pyratemaking.onleveling` — parallelogram method (W&M §5.2) and extension of exposures (W&M §5.3).
- `pyratemaking.trending` — multiplicative and additive trend fits for severity, frequency, and pure premium with OLS confidence intervals and a sensitivity-comparison helper.
- `pyratemaking.development` — chain ladder (Mack 1993), Bornhuetter-Ferguson, Cape Cod with decay, and four tail-factor methods (Bondy, Sherman, exponential decay, power curve). Bridge to `burncost` when installed.
- `pyratemaking.glm` — GLM adapter with `glum` and `statsmodels` backends sharing one API. Families: Poisson, Gamma, Tweedie, inverse Gaussian, binomial, Gaussian. Frequency-severity model, single-model Tweedie, penalised regression with cross-validated alpha, AIC/BIC stepwise selection, monotonicity-projected fitting. Optional GAM wrapper around `pygam`.
- `pyratemaking.core.indication` — loss-ratio (W&M Eq. 8.2) and pure-premium (Eq. 8.3) methods with credibility weighting against an outside complement.
- `pyratemaking.core.classification` — multi-way classification orchestrator with Tweedie or frequency-severity families and base-rate calibration.
- `pyratemaking.core.implementation` — caps, floors, dispersion summaries, segment-level summaries.
- `pyratemaking.relativities` — one-way and multi-way relativity tables, balance-principle check, credibility weighting, smoothing bridges to `actuarcredibility` and `whsmooth`.
- `pyratemaking.large_loss` — basic-limits losses, increased-limits factors, layer pricing (empirical and from `actudist` distributions).
- `pyratemaking.diagnostics` — lift, double-lift, decile lift, Gini and Lorenz, deviance and Pearson residuals, partial dependence and ALE, reliability diagrams.
- `pyratemaking.datasets` — French Motor TPL loader (Charpentier 2014, CC-BY-NC) with local cache, plus a deterministic synthetic generator that matches the same schema.
- `pyratemaking.reporting` — actuarial-styled tables, HTML rate filing template (Jinja2), Excel export with default number formats.
- Documentation site (mkdocs-material) with a quickstart and per-chapter user guide pages.
- Test suite with > 150 tests, including textbook reproductions of the Mack 1993 chain-ladder factors.
