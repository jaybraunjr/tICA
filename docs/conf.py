import importlib.metadata
import os
import sys

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "mdakit"
try:
    release = importlib.metadata.version("mdakit")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"

html_title = f"{project} {release}"
author = "R. Jay Braun Jr"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]
autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
