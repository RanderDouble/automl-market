PYTHON ?= python
export SOURCE_DATE_EPOCH := 1784131200

.PHONY: test experiment report slides all clean

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

experiment:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_experiments.py

report: experiment
	cd report && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex

slides: experiment
	cd slides && latexmk -xelatex -interaction=nonstopmode -halt-on-error midterm.tex

all: test report slides

clean:
	cd report && latexmk -C main.tex
	cd slides && latexmk -C midterm.tex
