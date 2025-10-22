# Makefile for Bend project

.PHONY: help install test lint format clean
.DEFAULT_GOAL := help

help:  ## Show this help message
	@echo ''
	@echo '$(BOLD)Bend - Data Analysis Tool$(RESET)'
	@echo ''
	@echo 'Usage:'
	@echo '  make $(CYAN)<target>$(RESET)'
	@echo ''
	@echo 'Targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ''

# Color definitions
CYAN := \033[36m
RESET := \033[0m
BOLD := \033[1m

install:  ## Install project dependencies
	pip install -e ".[dev]"
	npm install

test:  ## Run all tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=bend --cov-report=term-missing --cov-report=html

lint:  ## Run all linters (Python + Markdown)
	@echo "Running Python linting..."
	ruff check bend/ tests/
	@echo "\nRunning Markdown linting..."
	npm run lint:md

lint-py:  ## Run Python linter only
	ruff check bend/ tests/

lint-md:  ## Run Markdown linter only
	npm run lint:md

format:  ## Auto-format all code (Python + Markdown)
	@echo "Formatting Python code..."
	black bend/ tests/
	ruff check --fix bend/ tests/
	@echo "\nFormatting Markdown..."
	npm run lint:md:fix

format-py:  ## Auto-format Python code only
	black bend/ tests/
	ruff check --fix bend/ tests/

format-md:  ## Auto-format Markdown only
	npm run lint:md:fix

clean:  ## Clean up generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

