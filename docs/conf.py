import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath(".."))

project = "Causal-TS"
copyright = "2025-2026, Mohammad Fesanghary"
author = "Mohammad Fesanghary"
# Derive the version from the installed package so it never goes stale.
# NOTE: import the metadata helper under an alias — a bare ``version`` name
# would shadow Sphinx's own ``version`` config value with a function object
# and break the build (TypeError in inventory dump / smartquotes).
try:
    release = _pkg_version("causalts")
except PackageNotFoundError:
    release = "0.0.0"
version = release

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
    # Enables the ``{sub-ref}`today``` role used for the front-page
    # "Last updated" stamp, so the date tracks the build instead of
    # being hand-edited (and going stale).
    "substitution",
]

# Format for the front-page "Last updated" stamp -> e.g. "August 2026".
today_fmt = "%B %Y"

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

# The announcement banner is raw HTML injected verbatim into *every* page, so a
# relative href only resolves from the root document -- from examples/index.html
# it pointed at examples/examples/... and 404'd. The link target is filled in
# per page from ``pathto`` in ``_localize_announcement`` below; this literal is
# the root-relative fallback for the (unused) case where no context is present.
ANNOUNCEMENT_TARGET = "examples/agentic_discovery"
ANNOUNCEMENT_LEAD = (
    "🚀 New in v0.26 — the <code>causal-ts-discovery</code> agent skill, "
    "a <code>causal-ts inspect</code> pre-flight, and causal feature selection."
)
ANNOUNCEMENT_HTML = (
    ANNOUNCEMENT_LEAD + " <a href='{link}'>See the agentic workflow →</a>"
)

html_theme_options = {
    "announcement": ANNOUNCEMENT_HTML.format(link=f"{ANNOUNCEMENT_TARGET}.html"),
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
        # Match on the Read the Docs version *slug*, not the package version.
        # RTD serves exactly two versions here — /en/stable/ (built from the
        # release tag) and /en/latest/ (built from main) — so those slugs are
        # what switcher.json enumerates. Matching on ``release`` instead made
        # both builds report "0.26.0", which collapsed the dropdown to a single
        # entry with no route back from stable to latest.
        "version_match": os.environ.get("READTHEDOCS_VERSION") or "stable",
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


def _localize_announcement(app, pagename, templatename, context, doctree):
    """Rewrite the announcement link relative to the page being rendered.

    ``theme_announcement`` is dropped into the template as-is on every page, so
    the href has to be resolved per page rather than written once in the theme
    options.  ``pathto`` is the same helper the theme uses for its own links.
    """
    pathto = context.get("pathto")
    if pathto is None:
        return
    if pagename == ANNOUNCEMENT_TARGET:
        # Already there -- ``pathto`` would render a dead "#" self-link.
        context["theme_announcement"] = ANNOUNCEMENT_LEAD
        return
    context["theme_announcement"] = ANNOUNCEMENT_HTML.format(
        link=pathto(ANNOUNCEMENT_TARGET)
    )


def setup(app):
    app.connect("html-page-context", _localize_announcement)
