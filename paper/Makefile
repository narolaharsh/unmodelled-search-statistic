#       all - Will produce the main paper
#       clean - Will remove any intermediate products

NAME = unmodelled_search
BIBLIOGRAPHY = references.bib

PDF = $(NAME).pdf
TEXFILE = $(NAME).tex
SUPPORTING_TEXFILES = $(filter-out $TEXFILE, $(wildcard *.tex))

# Setting parameters for running of latexmk by the gitlab runner
LATEXMK = latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode"

#All Target is equivalent to making the PDF
all: $(PDF)

#PDF Target
$(PDF): $(TEXFILE) $(SUPPORTING_TEXFILES)
	$(LATEXMK) $(TEXFILE)

#Cleaning Target
.PHONY: clean
clean:
	rm -rf *.aux *.bbl *.blg *.log *.toc *.out *.fdb* *.fls gitID.txt


