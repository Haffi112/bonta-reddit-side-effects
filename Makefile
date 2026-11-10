.PHONY: install test lint typecheck fmt clean \
        inventory collect enrich extract map faers labels analyze \
        pipeline

install:
	uv sync
	uv run pre-commit install

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src/

fmt:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage

# Pipeline stages (each is a scripts/ entrypoint)
inventory:
	uv run python -m scripts.run_inventory

collect:
	uv run python -m scripts.run_collection

enrich:
	uv run python -m scripts.run_enrich

extract:
	uv run python -m scripts.run_extraction

map:
	uv run python -m scripts.run_mapping

faers:
	uv run python -m scripts.run_faers

labels:
	uv run python -m scripts.run_labels

analyze:
	uv run python -m scripts.run_analysis

# pipeline: collect → enrich → extract → ...
pipeline: collect enrich extract map faers labels analyze
	@echo "Pipeline complete."
