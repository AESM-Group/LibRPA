# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys
from docutils import nodes
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'LibRPA'
copyright = '2024, LibRPA authors'
html_show_copyright = False
author = 'LibRPA authors'

# -- General configuration ---------------------------------------------------
master_doc = "index"
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
extensions = ['.rst', '.md']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'theory/**',
                    "README.md", "*/README.md",]


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Sphinx's own extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # External stuff
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.doxylink",
    "sphinxcontrib.bibtex",
]
bibtex_bibfiles = ["references.bib"]

# -- Options for Doxylink -------------------------------------------------
doxygen_build_dir = "doxygen/librpa"
doxylink = {
    "librpa": (f"{doxygen_build_dir}/html/tagfile.xml", f"{doxygen_build_dir}/html"),
}

# -- Options for MyST output -------------------------------------------------
# myst_heading_anchors = 3
myst_heading_anchors = True
# Making Sphinx Support Markdown Math Formulas
myst_enable_extensions = [
    "dollarmath",  # Enable $ symbol for inline math
    "amsmath",     # Enable $$ symbol for display math
    "tasklist",    # Enable check list
    "attrs_inline",  # Enable inline attributes
    "colon_fence",   # Figure captions
]

numfig = True

# -- Options for todo extension -------------------------------------------------
# todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_book_theme'
# options for sphinx_book_theme
html_theme_options = {
  "show_toc_level": 3
}

html_title = "LibRPA Documentation"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    "custom.css",
]
html_js_files = [
    "open-dropdown.js",
]

# Configure MathJax
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}


def add_changelog_heading_ids(app, relative_path, parent_docname, content):
    """Give each release and its subsections stable, version-specific anchors."""
    if parent_docname != 'changelog' or str(relative_path) != '../CHANGELOG.md':
        return

    release_id = None
    lines = []
    for line in content[0].splitlines(keepends=True):
        release = re.match(r'^## \[([^\]]+)\]', line)
        subsection = re.match(r'^### (.+)', line)
        if release:
            version = release.group(1)
            release_id = nodes.make_id('v' + version) if version != 'Unreleased' else 'unreleased'
            lines.append(f'{{#{release_id}}}\n')
        elif subsection and release_id:
            section_id = nodes.make_id(subsection.group(1))
            lines.append(f'{{#{release_id}-{section_id}}}\n')
        lines.append(line)
    content[0] = ''.join(lines)


def setup(app):
    app.connect('include-read', add_changelog_heading_ids)
