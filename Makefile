.PHONY: install install-dev env check-env smoke test lint figures figures-demo results clean

# Intended workflow: conda activate rag3d (see scripts/setup_env.sh, README.md)
PYTHON ?= python
PIP ?= pip

install:
	$(PIP) install -e ".[dev,viz]"

install-dev: install

env:
	bash scripts/setup_env.sh

check-env:
	CUDA_VISIBLE_DEVICES= $(PYTHON) scripts/check_env.py

smoke:
	CUDA_VISIBLE_DEVICES= $(PYTHON) scripts/smoke_test.py

test:
	CUDA_VISIBLE_DEVICES= PYTHONPATH=src $(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m ruff check src scripts tests

figures:
	CUDA_VISIBLE_DEVICES= $(PYTHON) scripts/make_figures.py

figures-demo:
	CUDA_VISIBLE_DEVICES= $(PYTHON) scripts/make_figures.py --demo

results:
	$(PYTHON) scripts/collect_results.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
