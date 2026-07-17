PYTHON ?= python
export SOURCE_DATE_EPOCH := 1784131200

.PHONY: test experiment rq1 paper-experiments report slides rq-handout rq-slides all clean

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

experiment:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_experiments.py

rq1:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_rq1.py

paper-experiments:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_paper_experiments.py

report: experiment rq1 paper-experiments
	cd report && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex

slides: experiment rq1
	cd slides && latexmk -xelatex -interaction=nonstopmode -halt-on-error midterm.tex

rq-handout:
	cd docs && latexmk -xelatex -interaction=nonstopmode -halt-on-error rq1_rq3_handout.tex

rq-slides:
	cd slides/zby && latexmk -xelatex -interaction=nonstopmode -halt-on-error -output-directory=build C3_RQ1_RQ3.tex
	cp slides/zby/build/C3_RQ1_RQ3.pdf slides/rq1_rq3_presentation.pdf

all: test report slides

clean:
	cd report && latexmk -C main.tex
	cd slides && latexmk -C midterm.tex
	cd docs && latexmk -C rq1_rq3_handout.tex
	cd slides/zby && latexmk -C -output-directory=build C3_RQ1_RQ3.tex
