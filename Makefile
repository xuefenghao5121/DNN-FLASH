.PHONY: test lint run clean

test:
	python3 -m pytest -q

lint:
	python3 -m ruff check src tests

run:
	PYTHONPATH=src python3 -m paper_prototype.main

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
