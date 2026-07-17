PYTHON ?= python
export SOURCE_DATE_EPOCH := 1784131200

LATEX_BUILD_ROOT ?= /tmp/automl-market-latex
DELIVERABLE_DIR := deliverables

.PHONY: test experiment rq1 paper-experiments report slides \
	rq-handout deliverables all clean

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

experiment:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_experiments.py

rq1:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_rq1.py

paper-experiments:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_paper_experiments.py

report: experiment rq1 paper-experiments
	mkdir -p $(LATEX_BUILD_ROOT)/report $(DELIVERABLE_DIR)
	cd report && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/report main.tex
	cp $(LATEX_BUILD_ROOT)/report/main.pdf $(DELIVERABLE_DIR)/final_report.pdf

slides: experiment rq1 paper-experiments
	mkdir -p $(LATEX_BUILD_ROOT)/slides $(DELIVERABLE_DIR)
	cd slides && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/slides main.tex
	cp $(LATEX_BUILD_ROOT)/slides/main.pdf $(DELIVERABLE_DIR)/project_slides.pdf

rq-handout:
	mkdir -p $(LATEX_BUILD_ROOT)/rq-handout $(DELIVERABLE_DIR)
	cd docs && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/rq-handout rq1_rq3_handout.tex
	cp $(LATEX_BUILD_ROOT)/rq-handout/rq1_rq3_handout.pdf \
		$(DELIVERABLE_DIR)/rq1_rq3_handout.pdf

deliverables: report slides rq-handout

all: test deliverables

clean:
	rm -rf $(LATEX_BUILD_ROOT)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
