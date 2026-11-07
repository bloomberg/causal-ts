# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the rolling-statistic regime detector."""

import numpy as np
import pandas as pd
import pytest

from causalts.regime import ChangePointRegimeDetector


@pytest.mark.parametrize("stat", ["variance", "mean"])
def test_change_point_detector_assigns_every_row(stat):
    """``fillna(method="bfill")`` raised TypeError on pandas 3."""
    rng = np.random.default_rng(0)
    quiet = rng.standard_normal((200, 2))
    loud = 5.0 * rng.standard_normal((200, 2)) + (0.0 if stat == "variance" else 4.0)
    df = pd.DataFrame(np.vstack([quiet, loud]), columns=["a", "b"])

    labels = ChangePointRegimeDetector(n_regimes=2, window=40, stat=stat).assign(df)

    assert labels.shape == (400,) and labels.dtype == np.int32
    # Rows before min_periods (window // 2) are NaN and back-filled from the first
    # defined row, 19.
    first = 40 // 2 - 1
    assert (labels[:first] == labels[first]).all()
    assert labels[:150].mean() < 0.1 and labels[250:].mean() > 0.9
