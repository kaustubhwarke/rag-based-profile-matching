# Profile Matching — developer workflow shortcuts.
# Usage: make <target>   (on Windows, use `make` via Git Bash / WSL, or run the
# underlying commands directly — they are listed for each target).

PYTHON ?= python
export PYTHONPATH := src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD := 1

.PHONY: help install install-all data index match eval test lint format typecheck clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:         ## Install core (offline) package
	$(PYTHON) -m pip install -e .

install-all:     ## Install full production stack (HF + Chroma + loaders + dev)
	$(PYTHON) -m pip install -e ".[all]"

data:            ## Generate the sample dataset (32 résumés, 6 JDs, ground truth)
	$(PYTHON) scripts/generate_sample_data.py

index:           ## Part A — index résumés into the vector store
	$(PYTHON) resume_rag.py index --resume-dir data/resumes

match:           ## Part B — match the ML Engineer JD (example)
	$(PYTHON) job_matcher.py --jd-file data/job_descriptions/jd_01_ml_engineer.txt --top-k 10

eval:            ## Run the evaluation harness → outputs/evaluation.json
	$(PYTHON) scripts/run_evaluation.py

test:            ## Run the test suite
	$(PYTHON) -m pytest -q

lint:            ## Lint with ruff
	ruff check src tests

format:          ## Format with black + ruff --fix
	black src tests && ruff check --fix src tests

typecheck:       ## Static type-check with mypy
	mypy src

clean:           ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache .vectorstore outputs **/__pycache__ \
		src/*.egg-info
