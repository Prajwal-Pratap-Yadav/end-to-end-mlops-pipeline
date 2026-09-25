# Convenience targets (Linux/macOS). Every target is a thin wrapper around a
# plain command documented in the README, so Windows users can run those directly.
PYTHON ?= python

.PHONY: help install lint format typecheck test check train serve monitor simulate drift-demo \
        up down smoke logs clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install locked runtime + dev dependencies and git hooks
	$(PYTHON) -m pip install -r requirements-dev.txt
	pre-commit install

lint:  ## black, isort and ruff in check mode
	black --check .
	isort --check-only .
	ruff check .

format:  ## Auto-format the code base
	isort .
	black .
	ruff check --fix .

typecheck:  ## Strict mypy
	mypy

test:  ## Test suite with coverage
	$(PYTHON) -m pytest --cov --cov-report=term-missing

check: lint typecheck test  ## Everything CI runs before the Docker job

train:  ## Train, register and (if it passes the gate) promote a model
	$(PYTHON) -m src.train

serve:  ## Run the API on http://localhost:8000
	uvicorn app.main:app --host 0.0.0.0 --port 8000

monitor:  ## One monitoring cycle (drift, live metrics, retraining if triggered)
	$(PYTHON) -m monitoring.monitor --once

simulate:  ## Send normal traffic with delayed ground truth to the local API
	$(PYTHON) -m monitoring.simulate --n 1000

drift-demo:  ## Send traffic from a shifted market to the local API
	$(PYTHON) -m monitoring.simulate --n 1500 --drift-strength 0.8 --seed 8

up:  ## Build and start the full Docker stack
	docker compose up -d --build

down:  ## Stop the stack and delete its volumes
	docker compose --profile demo down -v

smoke:  ## Smoke-test every service of the running stack
	$(PYTHON) scripts/smoke_test.py --stack

logs:  ## Follow logs of the stack
	docker compose logs -f

clean:  ## Remove local MLflow store, generated data, reports and caches
	rm -rf mlflow.db mlruns mlartifacts data reports .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
