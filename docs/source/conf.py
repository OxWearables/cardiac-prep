"""Sphinx configuration.

The package is not installed into the docs environment. Importing it for real
would pull in TensorFlow, which is far larger than a documentation build has
any business downloading, so ``src`` goes on ``sys.path`` and the heavy
third-party imports are mocked for autodoc.
"""

import sys
from pathlib import Path

DOCS_SOURCE = Path(__file__).parent.resolve()
REPO_ROOT = DOCS_SOURCE.parents[1]

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(DOCS_SOURCE))

from _generate import write_configuration_page, write_datadict_page  # noqa: E402

# Generated here, at config-execution time, so the pages exist before Sphinx
# discovers source files. Doing it from setup() would be too late.
write_configuration_page(DOCS_SOURCE, REPO_ROOT / "config.yaml")
write_datadict_page(DOCS_SOURCE, REPO_ROOT / "docs" / "datadict.csv")

# -- Project information -----------------------------------------------------

project = "cardiac-prep"
copyright = "2026, University of Oxford"
author = "Anna Bator, Stefan van Duijvenboden"
release = "1.0.0"
version = "1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_generate.py"]

myst_enable_extensions = [
    "deflist",
    "colon_fence",
]
myst_heading_anchors = 3

# -- Autodoc -----------------------------------------------------------------

# Everything the package imports from outside the standard library, except
# PyYAML, which docs/requirements.txt installs because config.py needs it at
# import time to build the Config dataclass the configuration page reads.
autodoc_mock_imports = [
    "actipy",
    "keras",
    "matplotlib",
    "numpy",
    "pandas",
    "pyedflib",
    "reportlab",
    "scipy",
    "seaborn",
    "sklearn",
    "tensorflow",
]

autodoc_member_order = "bysource"
# Dataclasses generate an __init__ from their fields. Left alone, autodoc
# renders those fields once as constructor parameters and again as attributes.
# Separating the signature drops the constructor rendering and keeps the
# attribute list, which is the one that shows defaults.
autodoc_class_signature = "separated"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "cardiac-prep"
