.PHONY: install test lint typecheck check demo sweep bench

install:
	pip install -e ".[dev,bench]"

test:
	pytest -q

lint:
	ruff check .

typecheck:
	mypy gateway

check: lint typecheck test

demo:
	python -m gateway.demo

sweep:
	python -m benchmarks.threshold_sweep

bench:
	python -m benchmarks.bench_cache
