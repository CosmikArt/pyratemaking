# Contributing

Thanks for the interest. Issues and pull requests welcome.

## Workflow

1. Fork and clone.
2. `pip install -e ".[full,dev,docs]"` in a fresh venv.
3. `pre-commit install` before the first commit.
4. Make the change with tests.
5. `pytest` and `ruff check .` must pass locally.
6. Open a PR against `main` with a short description of what changed and why.

## Commits

Conventional commits in lowercase, terse:

```
feat(onleveling): parallelogram with non-uniform writing
fix(glm): exposure offset on tweedie
test(development): mack 1993 link factors
docs(indication): expense provision example
```

## Tests

New features need tests. For numerical methods that follow a published procedure, a textbook reproduction at `rtol=1e-4` is the bar. Mack 1993, Werner & Modlin worked examples, and Goldburd et al. 2020 examples are good targets.

## Style

- `ruff check` and `ruff format` are the source of truth.
- Type hints concise: `np.ndarray`, `pd.DataFrame`, `dict[str, float]`. Use `| None`, not `Optional[...]`.
- Docstrings: state the math, the actuarial meaning of each input, the citation, and any non-obvious gotcha.
- Variables named with actuarial vocabulary (`earned_exposure`, `incurred_losses`, `pure_premium`).
