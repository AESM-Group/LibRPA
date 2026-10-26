PDFLATEX ?= pdflatex
PDFTOPPM ?= pdftoppm
PLANTUML ?= plantuml
SOURCES := $(wildcard *.tex *.puml)
PNGS := $(sort $(addsuffix .png,$(basename $(SOURCES))))

.PHONY: all
.DELETE_ON_ERROR:

all: $(PNGS)

%.png: %.tex
	mkdir -p .build
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=.build "$<"
	$(PDFTOPPM) -r 300 -singlefile -png ".build/$*.pdf" "$*"

%.png: %.puml
	$(PLANTUML) -failfast2 -tpng -Sdpi=300 "$<"
