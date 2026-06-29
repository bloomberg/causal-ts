import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Causal-TS"
copyright = "2025-2026, Mohammad Fesanghary"
author = "Mohammad Fesanghary"
release = "0.14.0"

extensions = [
    "myst_nb",
    "sphinx_design",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
]

# MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "attrs_inline",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

# myst-nb: don't execute notebooks on build
nb_execution_mode = "off"

# sphinx-autoapi
autoapi_dirs = ["../causalts"]
autoapi_root = "api/autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autoapi_add_toctree_entry = False
suppress_warnings = ["autoapi.python_import_resolution"]

# Napoleon (numpy-style docstrings)
napoleon_numpy_docstring = True
napoleon_google_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

master_doc = "index"

# HTML output
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/img/cts_.png"
html_favicon = "_static/img/favicon_.png"
html_show_sourcelink = False

html_theme_options = {
    "announcement": "🚀 Causal-TS v0.17 released — CEDAR, partial_dcor lag selection, C-node nonstationarity support, and 20+ example notebooks. "
    "<a href='getting_started/index.html'>Get started →</a>",
    "logo": {
        "text": "Causal-TS",
        "image_light": "_static/img/cts_.png",
        "image_dark": "_static/img/cts_.png",
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-field"],
    "back_to_top_button": True,
    "check_switcher": False,
    "switcher": {
        "json_url": "_static/switcher.json",
        "version_match": release,
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/bloomberg/causal-ts",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/causalts",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "external_links": [],
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "show_nav_level": 2,
    "footer_start": ["copyright", "prev-next"],
    "footer_end": ["sphinx-version", "theme-version"],
    "primary_sidebar_end": [],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": True,
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}

html_sidebars = {
    "index": [],
    "getting_started/index": [],
}

html_context = {
    "github_user": "bloomberg",
    "github_repo": "causal-ts",
    "github_version": "main",
    "doc_path": "docs",
}
