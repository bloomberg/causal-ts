# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Guard the lazy-import surface.

``causalts.cedar.legacy`` (dcor, statsmodels) and ``causalts.grace``
(pytorch-lightning) are imported on demand to keep ``import causalts`` fast.
These tests pin both halves of that contract: the heavy modules stay unimported,
and every public name still resolves through every import form.

Laziness assertions must run in a fresh interpreter -- sibling test modules
import grace/legacy, which would pollute ``sys.modules`` in-process.
"""

import subprocess
import sys

import pytest

# "statsmodels" (the bare package), not just "statsmodels.api": the aggregator
# can stay unimported while a submodule import -- e.g. multivariate.cancorr --
# still drags the core package onto the import path.
HEAVY = ["dcor", "pytorch_lightning", "statsmodels"]
LAZY_MODULES = ["causalts.cedar.legacy", "causalts.grace"]
LAZY_NAMES = ["SYPI", "run_cdnots_gated", "run_stability_selection", "GraceResult"]


def _run(code):
    """Execute *code* in a clean interpreter, return stdout."""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.mark.parametrize("mod", HEAVY + LAZY_MODULES)
def test_heavy_module_not_imported_eagerly(mod):
    """A plain ``import causalts`` must not drag in the expensive deps."""
    loaded = _run(f"import sys, causalts; print({mod!r} in sys.modules)")
    assert loaded == "False", f"{mod} was imported eagerly by `import causalts`"


@pytest.mark.parametrize("name", LAZY_NAMES)
def test_lazy_name_resolves_by_attribute(name):
    assert _run(f"import causalts; print(causalts.{name}.__name__)")


@pytest.mark.parametrize("name", LAZY_NAMES)
def test_lazy_name_resolves_by_from_import(name):
    assert _run(f"from causalts import {name}; print({name}.__name__)")


@pytest.mark.parametrize("name", LAZY_NAMES)
def test_lazy_name_exported_by_star_import(name):
    """Regression: __getattr__ alone does not feed ``from causalts import *``."""
    assert _run(f"exec('from causalts import *'); print({name}.__name__)")


@pytest.mark.parametrize("sub", ["grace", "plotting", "cedar"])
def test_submodule_attribute_access(sub):
    """Regression: eager imports used to bind these as package attributes."""
    assert _run(f"import causalts; print(causalts.{sub}.__name__)") == f"causalts.{sub}"


def test_cedar_legacy_attribute_access():
    assert (
        _run("import causalts; print(causalts.cedar.legacy.__name__)")
        == "causalts.cedar.legacy"
    )


def test_direct_submodule_import_still_works():
    assert _run("import causalts.grace.gated_discovery as g; print(g.__name__)")


def test_import_submodule_before_package():
    """Importing a lazy submodule first must not deadlock or double-register."""
    assert _run(
        "import causalts.grace; import causalts; "
        "from causalts.algorithms import list_algorithms; print(list_algorithms())"
    )


def test_grace_registered_without_importing_lightning():
    """The registry lists grace lazily; listing must not pay the import cost."""
    out = _run(
        "import sys; from causalts.algorithms import list_algorithms; "
        "names = list_algorithms(); "
        "print(('grace' in names) and ('grace-ss' in names), "
        "'pytorch_lightning' in sys.modules)"
    )
    assert out == "True False"


def test_unknown_attribute_raises_attribute_error():
    import causalts

    with pytest.raises(AttributeError, match="has no attribute"):
        causalts.definitely_not_a_thing


def test_unknown_algorithm_still_raises_value_error():
    from causalts.algorithms import run_algorithm

    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_algorithm("bogus", None, None, 1)


def test_dir_lists_lazy_names():
    import causalts

    listed = dir(causalts)
    for name in LAZY_NAMES + ["grace"]:
        assert name in listed, f"{name} missing from dir(causalts)"
