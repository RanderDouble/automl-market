PYTHON ?= python
export SOURCE_DATE_EPOCH := 1784131200

.PHONY: test experiment rq1 report slides all clean

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

experiment:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_experiments.py

rq1:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_rq1.py

report: experiment rq1
	cd report && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex

slides: experiment rq1
	cd slides && latexmk -xelatex -interaction=nonstopmode -halt-on-error midterm.tex

all: test report slides

clean:
	cd report && latexmk -C main.tex
	cd slides && latexmk -C midterm.tex
