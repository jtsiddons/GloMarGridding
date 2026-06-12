.PHONY: clean clean-build clean-pyc clean-test coverage docs help install lint test dev
.DEFAULT_GOAL := help

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-docs: ## remove docs artifacts
	rm -rf docs/_build
	rm docs/Documentation.{tex,pdf}

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr .pytest_cache
	rm -fr htmlcov/

install:
	python -m pip install .

install-docs: ## install dependencies needed for building the docs
	python -m pip install --quiet --group docs

install-test: ## install dependencies needed for standard testing
	python -m pip install --quiet --group test

install-lint: ## install dependencies needed for linting
	python -m pip install --quiet --group lint

lint: install-lint ## check style
	python -m ruff check glomar_gridding test
	python -m numpydoc lint glomar_gridding/**.py
	codespell glomar_gridding test docs

test: install install-test ## run tests quickly with the default Python
	python -m pytest

coverage: install-test ## check code coverage quickly with the default Python
	python -m coverage run --source glomar_gridding -m pytest
	python -m coverage report -m
	python -m coverage html

docs: clean-docs install install-docs ## generate Sphinx HTML documentation, including API docs
	sphinx-build -M latex ./docs ./docs/_build
	make -C docs/_build/latex
	make -C docs/_build/latex
	cp docs/_build/latex/glomargridding.tex docs/Documentation.tex
	cp docs/_build/latex/glomargridding.pdf docs/Documentation.pdf

dev: clean ## install the package to the active Python's site-packages
	python -m pip install --group dev
	python -m pip install --no-user --editable .
	pre-commit install
