# Contributing

Thanks for your interest in improving this project. This guide covers the local
workflow and the standards every change is held to.

## Development setup

Requires Python 3.11+ (3.11 and 3.12 are tested in CI) and, for the full stack,
Docker with Compose v2.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install                 # runs isort, black, ruff and mypy on every commit
```

## Quality gates

CI runs these on every push and pull request; run them locally first:

| Check | Command |
|-------|---------|
| Formatting | `black --check .` and `isort --check-only .` |
| Linting | `ruff check .` |
| Types (strict) | `mypy` |
| Tests + coverage | `pytest --cov` |
| Full stack | `docker compose up -d --build` then `python scripts/smoke_test.py --stack` |

`make check` runs the first four in one go (Linux/macOS).

## Standards

- **Types and docstrings.** Every function is fully type-annotated (`mypy --strict`
  passes) and public functions carry Google-style docstrings.
- **Tests.** New behaviour ships with tests. Prefer real components over mocks:
  the fixtures in `tests/conftest.py` train small models against a throwaway
  MLflow registry in a temporary directory.
- **One feature contract.** Input features are defined once in `src/schema.py`.
  Adding or changing a feature means updating the contract; the API schema,
  validation, preprocessing and drift detection derive from it.
- **Configuration, not constants.** Tunables belong in `configs/config.yaml`,
  validated by the models in `src/config.py`.
- **No secrets in the repository.** Use environment variables (see `.env.example`).

## Dependencies

Dependencies are declared with compatible ranges in `pyproject.toml` and pinned in
lock files generated with [uv](https://docs.astral.sh/uv/):

```bash
uv pip compile pyproject.toml --universal --python-version 3.11 -o requirements.txt
uv pip compile pyproject.toml --extra dev --universal --python-version 3.11 -o requirements-dev.txt
```

Commit the regenerated lock files together with the `pyproject.toml` change, and
keep the hook versions in `.pre-commit-config.yaml` in sync with the dev pins.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/):
  `feat(monitoring): ...`, `fix(api): ...`, `docs: ...`, `test: ...`, `build: ...`.
- Keep pull requests focused; describe the motivation and how you verified the change.
- CI must be green: lint, type-check, the test matrix and the Docker end-to-end job.
