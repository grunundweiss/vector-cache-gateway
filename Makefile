.PHONY: install install-all test lint typecheck check demo sweep bench ann guard intent serve

install:
	pip install -e ".[encoder,dev,bench]"

install-all:
	pip install -e ".[encoder,dev,bench,ann,serve]"

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

intent:
	python -m benchmarks.intent_sweep

guard:
	python -m benchmarks.guard_eval

bench:
	python -m benchmarks.bench_cache

ann:
	python -m benchmarks.bench_ann

serve:
	uvicorn examples.fastapi_proxy:app --port 8000
