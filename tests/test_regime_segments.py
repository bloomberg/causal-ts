# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""``extract_segments`` applies ``min_length`` to the regime run, not the buffer."""

import numpy as np
import pandas as pd
import pytest

from causalts.regime.segments import extract_segments


def test_short_run_in_the_middle_is_rejected_despite_buffer():
    """Two regime rows plus a buffer of four used to pass ``min_length=5``.

    The prepended rows are lag context from the previous regime.
    """
    labels = np.array([0] * 8 + [1] * 2 + [0] * 6)
    df = pd.DataFrame({"X": range(len(labels))})
    assert extract_segments(df, labels, regime=1, min_length=5, buffer=4) == []


def test_short_run_at_the_end_is_rejected_despite_buffer():
    """A short run that reaches the end of the series is held to the same rule."""
    labels = np.array([0] * 8 + [1] * 2)
    df = pd.DataFrame({"X": range(len(labels))})
    assert extract_segments(df, labels, regime=1, min_length=5, buffer=4) == []


@pytest.mark.parametrize("tail", [0, 3], ids=["end", "middle"])
def test_run_of_exactly_min_length_is_accepted(tail):
    """A run of exactly ``min_length`` rows is kept, and the buffer is still prepended."""
    prefix = 6
    run = 5
    buffer = 4
    labels = np.array([0] * prefix + [1] * run + [0] * tail)
    df = pd.DataFrame({"X": range(len(labels))})
    segs = extract_segments(df, labels, regime=1, min_length=run, buffer=buffer)
    assert len(segs) == 1
    seg = segs[0]
    buf_start = prefix - buffer
    end = prefix + run
    assert seg.regime == 1
    assert seg.start_idx == buf_start
    assert seg.end_idx == end
    assert int((labels[seg.start_idx : seg.end_idx] == 1).sum()) == run
    assert len(seg.df) == run + buffer
    pd.testing.assert_frame_equal(seg.df, df.iloc[buf_start:end])
