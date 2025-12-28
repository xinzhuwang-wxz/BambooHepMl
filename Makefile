# Makefile
SHELL = /bin/bash

# Styling
.PHONY: style
style:
	black bamboohepml tests
	flake8 bamboohepml tests
	python3 -m isort bamboohepml tests
	pyupgrade --py39-plus bamboohepml/**/*.py tests/**/*.py

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage*
	rm -rf htmlcov/
	rm -rf site/

# Install
.PHONY: install
install:
	pip install -e ".[test,serve]"

# Test
.PHONY: test
test:
	pytest tests/ -v

# Test with coverage
.PHONY: test-cov
test-cov:
	pytest tests/ -v --cov=bamboohepml --cov-report=html --cov-report=term

# Lint
.PHONY: lint
lint:
	flake8 bamboohepml tests
	black --check bamboohepml tests
	isort --check-only bamboohepml tests

# Docs
.PHONY: docs
docs:
	mkdocs build

# Serve docs
.PHONY: docs-serve
docs-serve:
	mkdocs serve

