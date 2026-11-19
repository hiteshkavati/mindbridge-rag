.PHONY: install test lint eval serve kb-check

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests

eval:
	mindbridge eval --backend lsa --write docs/eval-report.md

kb-check:
	mindbridge kb-check

serve:
	mindbridge serve
