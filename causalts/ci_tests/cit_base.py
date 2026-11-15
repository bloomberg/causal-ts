# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Deprecated shim: use :class:`causalts.ci_tests.cit_test.CIT_Base` instead."""

import warnings

from .cit_test import CIT_Base  # noqa: F401

warnings.warn(
    "cit_base.CIT_Base is deprecated and consolidated into cit_test.CIT_Base. "
    "Update imports to: from causalts.ci_tests.cit_test import CIT_Base",
    DeprecationWarning,
    stacklevel=2,
)
