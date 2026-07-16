PYTHON ?= python

.PHONY: test experiment report all clean

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

experiment:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_experiments.py

report: experiment
	cd report && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex

all: test report

clean:
	cd report && latexmk -C main.tex

