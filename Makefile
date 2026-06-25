.PHONY: test test-python test-cpp build bench lint clean

build:
	cmake -S . -B build
	cmake --build build -j

test-python:
	PYTHONPATH=python python3 -m pytest -q tests/python

test-cpp: build
	ctest --test-dir build --output-on-failure

test: test-python test-cpp

bench: build
	./build/flashone_bench

lint:
	python3 -m ruff check python tests/python

clean:
	rm -rf build .pytest_cache .ruff_cache python/*.egg-info *.egg-info
