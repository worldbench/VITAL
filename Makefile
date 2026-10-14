PYTHON ?= python

.PHONY: test lint audit check build release
test:
	$(PYTHON) -m unittest discover -s tests -v

lint:
	$(PYTHON) -m ruff check src tests tools

audit:
	$(PYTHON) tools/release.py

check: lint test audit

build:
	$(PYTHON) tools/build.py

release: check
	$(PYTHON) tools/release.py --output dist/vital-source.zip
