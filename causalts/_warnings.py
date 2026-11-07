# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Warning helpers shared across causalts."""

import os
import sys

_PKG = os.path.dirname(os.path.abspath(__file__)) + os.sep


def external_stacklevel():
    """``stacklevel`` that points ``warnings.warn`` at the first caller outside causalts.

    Call it in the ``warnings.warn(...)`` call itself. Library entry points reach
    the same warning at different depths (e.g. ``run_cdnots`` →
    ``cdnots_discovery``, or ``Cedar`` / ``run_cedar`` / ``run_algorithm``), so a
    fixed level lands inside the library on some paths. Python's default filters
    also show a ``DeprecationWarning`` only when it is attributed to ``__main__``.
    Replaceable by ``warnings.warn(..., skip_file_prefixes=...)`` once Python 3.12
    is the minimum.
    """
    frame, level = sys._getframe(1), 1  # level 1: the function calling warn
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(
        _PKG
    ):
        frame, level = frame.f_back, level + 1
    return level
