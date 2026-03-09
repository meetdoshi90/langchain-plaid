.PHONY: test lint lint_package lint_tests format build clean

test:
	poetry run pytest tests/unit_tests -v

lint_package:
	poetry run ruff check langchain_plaid
	poetry run mypy langchain_plaid

lint_tests:
	poetry run ruff check tests

lint: lint_package lint_tests

format:
	poetry run ruff format langchain_plaid tests
	poetry run ruff check --fix langchain_plaid tests

build:
	poetry build

clean:
	rm -rf dist/ .ruff_cache/ .mypy_cache/ __pycache__/ langchain_plaid/__pycache__/