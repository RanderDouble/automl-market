PYTHON ?= python
export SOURCE_DATE_EPOCH := 1784131200
export XDG_CACHE_HOME ?= /tmp/automl-market-cache

LATEX_BUILD_ROOT ?= /tmp/automl-market-latex
DELIVERABLE_DIR := deliverables

.PHONY: test experiment legacy-experiment improvements rq1 paper-experiments results report slides \
	rq-handout improvement-report paper-summary deliverables all clean

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

experiment: legacy-experiment

legacy-experiment:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_experiments.py

improvements:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_improvement_experiments.py

rq1:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_rq1.py

paper-experiments:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_paper_experiments.py

results: experiment improvements rq1 paper-experiments

report:
	mkdir -p $(LATEX_BUILD_ROOT)/report $(DELIVERABLE_DIR)
	cd report && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/report main.tex
	cp $(LATEX_BUILD_ROOT)/report/main.pdf $(DELIVERABLE_DIR)/final_report.pdf

slides:
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

improvement-report:
	mkdir -p $(LATEX_BUILD_ROOT)/improvement-report $(DELIVERABLE_DIR)
	cd docs && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/improvement-report \
		improvement_technical_report.tex
	cp $(LATEX_BUILD_ROOT)/improvement-report/improvement_technical_report.pdf \
		$(DELIVERABLE_DIR)/improvement_technical_report.pdf

paper-summary:
	mkdir -p $(LATEX_BUILD_ROOT)/paper-summary
	cd papers && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/paper-summary \
		Han2023_Optimal_Pricing_Data-Augmented_AutoML_summary.tex
	cp $(LATEX_BUILD_ROOT)/paper-summary/Han2023_Optimal_Pricing_Data-Augmented_AutoML_summary.pdf \
		papers/Han2023_Optimal_Pricing_Data-Augmented_AutoML_summary.pdf

deliverables: report slides rq-handout improvement-report paper-summary

all: test results deliverables

clean:
	rm -rf $(LATEX_BUILD_ROOT)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
