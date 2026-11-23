# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""``PeltRegimeDetector`` must not swallow misspelled keyword arguments."""

import pytest

from causalts.regime import PeltRegimeDetector


def test_unknown_keyword_argument_raises():
    """``PeltRegimeDetector(**kwargs)`` used to discard what it did not know."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        PeltRegimeDetector(penalty=1, min_seg=10)


def test_documented_arguments_still_work():
    detector = PeltRegimeDetector(pen=1.0, min_size=10, n_regimes=2, signal="mean")

    assert (detector.pen, detector.min_size, detector.n_regimes) == (1.0, 10, 2)
