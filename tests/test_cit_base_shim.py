# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the deprecated ``causalts.ci_tests.cit_base`` shim."""

import importlib
import sys

import pytest


def test_cit_base_warns_and_reexports_cit_test_class():
    from causalts.ci_tests.cit_test import CIT_Base as Canonical

    sys.modules.pop("causalts.ci_tests.cit_base", None)
    with pytest.warns(DeprecationWarning, match="cit_test.CIT_Base"):
        mod = importlib.import_module("causalts.ci_tests.cit_base")
    assert mod.CIT_Base is Canonical
