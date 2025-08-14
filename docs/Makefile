# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Get the parent dir name which is this docs' version
VERSION := $(notdir $(CURDIR))

github:
	@rm -rf build/html build/doctrees source/_autosummary/*.rst
	@cp -rf ../../assets/* source/_static/
	@cp -rf ../../examples/* source/_notebooks/examples/
	@make html
	@mkdir -p ../../docs/$(VERSION)
	@rm -rf ../../docs/$(VERSION)/*
	@cp -a build/html/. ../../docs/$(VERSION)/
	@cp ../versions.json ../../docs/versions.json

local:
	@python -m http.server --directory ../../docs/$(VERSION)/ 8080
