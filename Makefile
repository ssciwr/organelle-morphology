# Makefile — run tests and the marimo app against THIS checkout.
#
# The package is a namespace package installed editable elsewhere, so we put
# this checkout's src/ first on PYTHONPATH to test/run the local code.
#
# Uses the conda env named below by default (no activation needed):
#   make test        full test suite
#   make app         launch om_app (marimo) from this checkout
#
# Overrides:
#   make test ENV=other-env          use a different conda env
#   make test RUN=                   use the already-activated python
#   make test PYTEST_ARGS="-k hash"  pass extra pytest args

CONDA  ?= conda
ENV    ?= morph
PYTHON ?= python
RUN    ?= $(CONDA) run --no-capture-output -n $(ENV)

SRC := $(CURDIR)/src
export PYTHONPATH := $(SRC)$(if $(PYTHONPATH),:$(PYTHONPATH),)

.DEFAULT_GOAL := help
.PHONY: help test app

help:
	@echo "make test        - full test suite (conda env: $(ENV))"
	@echo "make app         - launch om_app (marimo) from this checkout"
	@echo "overrides: ENV=<env>  RUN=  PYTEST_ARGS=..."

test:
	$(RUN) $(PYTHON) -m pytest $(PYTEST_ARGS)

app:
	$(RUN) $(PYTHON) -m organelle_morphology.cli.om_app
