REPORT=internship_report
PDF_OUT=../../output/pdf/roadside_pm_internship_report.pdf

.PHONY: all figures clean

all: figures
	mkdir -p ../../output/pdf
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error $(REPORT).tex; \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic $(REPORT).tex; \
	else \
		pdflatex -interaction=nonstopmode -halt-on-error $(REPORT).tex && \
		pdflatex -interaction=nonstopmode -halt-on-error $(REPORT).tex; \
	fi
	cp $(REPORT).pdf $(PDF_OUT)

figures:
	python3 generate_figures.py

clean:
	rm -f $(REPORT).aux $(REPORT).bbl $(REPORT).blg $(REPORT).fdb_latexmk
	rm -f $(REPORT).fls $(REPORT).lof $(REPORT).log $(REPORT).lot
	rm -f $(REPORT).out $(REPORT).toc $(REPORT).pdf
