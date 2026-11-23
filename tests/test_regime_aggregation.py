# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for regime graph aggregation."""

import numpy as np
import pytest

from causalts.regime.aggregation import aggregate_majority


@pytest.mark.parametrize(
    ("n_regimes", "n_votes", "threshold", "kept"),
    [(25, 7, 0.28, 1), (25, 6, 0.28, 0), (6, 5, 5 / 6, 1), (6, 4, 5 / 6, 0)],
)
def test_majority_threshold_boundary(n_regimes, n_votes, threshold, kept):
    graphs = {
        regime: np.array([[[int(regime < n_votes)]]], dtype=np.int8)
        for regime in range(n_regimes)
    }

    result = aggregate_majority(graphs, threshold=threshold)

    assert result[0, 0, 0] == kept
