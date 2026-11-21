# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for ``temporal_bootstrap``."""

import numpy as np
import pandas as pd

from causalts import temporal_bootstrap


def test_each_run_keeps_its_own_output_when_discovery_fn_reuses_a_buffer():
    buffer = np.zeros((1, 1, 2))
    count = [0]

    def reuse_output(window):
        count[0] += 1
        buffer[0, 0, 1] = count[0]
        return buffer

    r = temporal_bootstrap(
        pd.DataFrame({"X": range(30)}),
        reuse_output,
        n_bootstrap=3,
        window_frac=0.5,
        show_progress=False,
    )

    assert [float(g[0, 0, 1]) for g in r["graphs"]] == [1.0, 2.0, 3.0]
    assert float(r["persistence"][0, 0, 1]) == 2.0
