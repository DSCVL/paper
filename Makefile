default: all

all:
	pdflatex *.tex

clean:
	rm -f *.ps *.dvi *.out *.log *.aux *.bbl *.blg *.pyg *.toc *.pdf
