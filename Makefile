PYTHON ?= python3
VENV := .venv

.PHONY: smoke
smoke:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install .
	$(VENV)/bin/python -m almm_harness --smoke
