PYTHON ?= python
export SOURCE_DATE_EPOCH := 1784131200

LATEX_BUILD_ROOT ?= /tmp/automl-market-latex
DELIVERABLE_DIR := deliverables

.PHONY: test experiment rq1 paper-experiments paper-summary report slides \
	rq-handout rq-slides rq1-slides rq3-slides zby-slides deliverables all clean

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

experiment:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_experiments.py

rq1:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_rq1.py

paper-experiments:
	MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src $(PYTHON) scripts/run_paper_experiments.py

paper-summary:
	mkdir -p $(LATEX_BUILD_ROOT)/paper-summary $(DELIVERABLE_DIR)
	cd papers && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/paper-summary \
		Han2023_Optimal_Pricing_Data-Augmented_AutoML_summary.tex
	cp $(LATEX_BUILD_ROOT)/paper-summary/Han2023_Optimal_Pricing_Data-Augmented_AutoML_summary.pdf \
		papers/Han2023_Optimal_Pricing_Data-Augmented_AutoML_summary.pdf

report: experiment rq1 paper-experiments
	mkdir -p $(LATEX_BUILD_ROOT)/report $(DELIVERABLE_DIR)
	cd report && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/report main.tex
	cp $(LATEX_BUILD_ROOT)/report/main.pdf $(DELIVERABLE_DIR)/final_report.pdf

# Earlier complete-project deck. The focused RQ1/RQ3 deck is built by rq-slides.
slides: experiment rq1
	mkdir -p $(LATEX_BUILD_ROOT)/project-overview $(DELIVERABLE_DIR)
	cd slides/project_overview && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/project-overview main.tex
	cp $(LATEX_BUILD_ROOT)/project-overview/main.pdf \
		$(DELIVERABLE_DIR)/project_overview_slides.pdf

rq-handout:
	mkdir -p $(LATEX_BUILD_ROOT)/rq-handout $(DELIVERABLE_DIR)
	cd docs && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/rq-handout rq1_rq3_handout.tex
	cp $(LATEX_BUILD_ROOT)/rq-handout/rq1_rq3_handout.pdf \
		$(DELIVERABLE_DIR)/rq1_rq3_handout.pdf

# Compile from the ZBY directory so its style keeps resolving figures/background.png.
# The wrapper and all RQ1/RQ3 content remain physically separate in slides/rq1_rq3/.
rq-slides:
	mkdir -p $(LATEX_BUILD_ROOT)/rq-slides $(DELIVERABLE_DIR)
	cd slides/zby && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/rq-slides ../rq1_rq3/main.tex
	cp $(LATEX_BUILD_ROOT)/rq-slides/main.pdf $(DELIVERABLE_DIR)/rq1_rq3_slides.pdf

zby-slides:
	mkdir -p $(LATEX_BUILD_ROOT)/zby-slides $(DELIVERABLE_DIR)
	cd slides/zby && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/zby-slides C3_Paper_Theory.tex
	cp $(LATEX_BUILD_ROOT)/zby-slides/C3_Paper_Theory.pdf \
		$(DELIVERABLE_DIR)/zby_theory_slides.pdf

deliverables: report slides rq-handout rq-slides zby-slides paper-summary

all: test deliverables

clean:
	rm -rf $(LATEX_BUILD_ROOT)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache

rq1-slides:
	mkdir -p $(LATEX_BUILD_ROOT)/rq1-slides $(DELIVERABLE_DIR)
	cd slides/zby && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/rq1-slides ../rq1_rq3/rq1_main.tex
	cp $(LATEX_BUILD_ROOT)/rq1-slides/rq1_main.pdf $(DELIVERABLE_DIR)/rq1_slides.pdf

rq3-slides:
	mkdir -p $(LATEX_BUILD_ROOT)/rq3-slides $(DELIVERABLE_DIR)
	cd slides/zby && latexmk -xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(LATEX_BUILD_ROOT)/rq3-slides ../rq1_rq3/rq3_main.tex
	cp $(LATEX_BUILD_ROOT)/rq3-slides/rq3_main.pdf $(DELIVERABLE_DIR)/rq3_slides.pdf
