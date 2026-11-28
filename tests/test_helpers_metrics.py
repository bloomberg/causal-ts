# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the set metrics, ``precision_recall_3d`` and logging helpers in
``causalts.utils.helpers``."""

import logging

import numpy as np
import pytest

from causalts.utils.helpers import (
    LogFormatter,
    precision,
    precision_recall,
    precision_recall_3d,
    recall,
    set_up_logging,
)

# --------------------------------------------------------------------------
# precision / recall / precision_recall
# --------------------------------------------------------------------------


def test_precision_and_recall_hand_computed():
    truth = {1, 2, 3, 4}
    pred = [3, 4, 5]
    assert precision(truth, pred) == pytest.approx(2 / 3)
    assert recall(truth, pred) == pytest.approx(2 / 4)


def test_precision_empty_prediction_is_zero():
    assert precision({1, 2}, []) == 0


def test_recall_empty_truth_is_nan():
    assert np.isnan(recall(set(), [1]))


def test_precision_recall_argument_order():
    # precision_recall(a, b) == (precision(a, b), recall(a, b)):
    # |a & b| / |b| and |a & b| / |a|.
    a, b = {1, 2}, {2, 3, 4, 5}
    p, r = precision_recall(a, b)
    assert p == pytest.approx(1 / 4)
    assert r == pytest.approx(1 / 2)


# --------------------------------------------------------------------------
# precision_recall_3d
# --------------------------------------------------------------------------


def test_precision_recall_3d_ignores_lag1_self_loops_in_truth():
    gt = np.zeros((3, 3, 2), dtype=int)
    gt[0, 1, 1] = 1
    gt[1, 2, 0] = 1
    gt[0, 0, 1] = 1  # AR self-loop: dropped from truth
    gt[2, 2, 1] = 1  # AR self-loop: dropped from truth
    est = np.zeros_like(gt)
    est[0, 1, 1] = 1  # TP
    est[2, 0, 1] = 1  # FP
    est[0, 0, 1] = 1  # self-loop not in (trimmed) truth -> FP
    p, r = precision_recall_3d(gt, est)
    assert p == pytest.approx(1 / 3)
    assert r == pytest.approx(1 / 2)
    # The caller's ground truth is not modified.
    assert gt[0, 0, 1] == 1 and gt[2, 2, 1] == 1


def test_precision_recall_3d_empty_estimate_has_zero_precision():
    gt = np.zeros((2, 2, 2), dtype=int)
    gt[0, 1, 1] = 1
    est = np.zeros_like(gt)
    p, r = precision_recall_3d(gt, est)
    assert p == 0.0
    assert r == 0.0


def test_precision_recall_3d_perfect():
    gt = np.zeros((3, 3, 3), dtype=int)
    gt[0, 1, 2] = gt[1, 2, 0] = 1
    p, r = precision_recall_3d(gt, gt.copy())
    assert (p, r) == (1.0, 1.0)


def test_precision_recall_3d_bool_graphs_match_int_graphs():
    gt = np.zeros((3, 3, 2), dtype=bool)
    gt[0, 1, 1] = gt[1, 2, 0] = True
    est = np.zeros_like(gt)
    est[0, 1, 1] = True  # TP
    est[2, 0, 1] = True  # FP
    expected = precision_recall_3d(gt.astype(int), est.astype(int))
    assert expected == (pytest.approx(1 / 2), pytest.approx(1 / 2))
    assert precision_recall_3d(gt, est) == expected
    assert precision_recall_3d(gt, gt.copy()) == (1.0, 1.0)


def test_precision_recall_3d_lag0_only_graph():
    gt = np.zeros((3, 3, 1), dtype=int)
    gt[0, 1, 0] = gt[1, 2, 0] = 1
    est = np.zeros_like(gt)
    est[0, 1, 0] = 1
    p, r = precision_recall_3d(gt, est)
    assert p == 1.0
    assert r == pytest.approx(1 / 2)


@pytest.mark.filterwarnings("error")
def test_precision_recall_3d_empty_truth_empty_estimate():
    gt = np.zeros((3, 3, 2), dtype=int)
    est = np.zeros_like(gt)
    p, r = precision_recall_3d(gt, est)
    assert (p, r) == (0.0, 0.0)


@pytest.mark.filterwarnings("error")
def test_precision_recall_3d_empty_truth_nonempty_estimate():
    gt = np.zeros((3, 3, 2), dtype=int)
    est = np.zeros_like(gt)
    est[0, 1, 1] = 1  # FP: truth has no links
    p, r = precision_recall_3d(gt, est)
    assert (p, r) == (0.0, 0.0)


@pytest.mark.filterwarnings("error")
def test_precision_recall_3d_only_lag1_self_loops_in_truth():
    gt = np.zeros((3, 3, 2), dtype=int)
    gt[0, 0, 1] = gt[1, 1, 1] = 1  # AR self-loops: dropped, so truth is empty
    est = np.zeros_like(gt)
    est[0, 1, 1] = 1
    p, r = precision_recall_3d(gt, est)
    assert (p, r) == (0.0, 0.0)


# --------------------------------------------------------------------------
# LogFormatter / set_up_logging
# --------------------------------------------------------------------------

_TEMPLATE = "%(color_on)s[%(levelname)s] %(message)s%(color_off)s"


def _record(level, msg="hello"):
    return logging.LogRecord("t", level, __file__, 1, msg, None, None)


def test_log_formatter_color_on_wraps_known_levels():
    fmt = LogFormatter(color=True, fmt=_TEMPLATE)
    out = fmt.format(_record(logging.ERROR))
    assert out == "\033[1;31m[ERROR] hello\033[0m"
    out = fmt.format(_record(logging.WARNING))
    assert out.startswith(LogFormatter.COLOR_CODES[logging.WARNING])


def test_log_formatter_no_color_or_unknown_level_is_plain():
    assert LogFormatter(color=False, fmt=_TEMPLATE).format(_record(logging.ERROR)) == (
        "[ERROR] hello"
    )
    # Level 25 has no color code even when color is on.
    rec = _record(25)
    assert LogFormatter(color=True, fmt=_TEMPLATE).format(rec) == ("[Level 25] hello")


@pytest.fixture
def restore_logging():
    root_handlers = list(logging.root.handlers)
    created = []
    yield created
    for name in created:
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            h.close()
            lg.removeHandler(h)
    logging.root.handlers = root_handlers


def _setup(name, tmp_path, **overrides):
    kwargs = dict(
        logger_name=name,
        console_log_output="stdout",
        console_log_level="warning",
        console_log_color=False,
        logfile_file=str(tmp_path / "run.log"),
        logfile_log_level="debug",
        logfile_log_color=False,
        log_line_template="[%(levelname)s] %(message)s",
    )
    kwargs.update(overrides)
    return set_up_logging(**kwargs)


def test_set_up_logging_routes_levels_to_console_and_file(
    tmp_path, capsys, restore_logging
):
    restore_logging.append("causalts_test_log_a")
    logger = _setup("causalts_test_log_a", tmp_path)
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.DEBUG
    assert logging.root.handlers == []
    assert len(logger.handlers) == 2

    logger.debug("dbg")
    logger.warning("warn")
    for h in logger.handlers:
        h.flush()

    out = capsys.readouterr().out
    assert "[WARNING] warn" in out
    assert "dbg" not in out  # console level is WARNING
    text = (tmp_path / "run.log").read_text()
    assert "[DEBUG] dbg" in text and "[WARNING] warn" in text


def test_set_up_logging_stderr_and_overwrite_mode(tmp_path, capsys, restore_logging):
    (tmp_path / "run.log").write_text("old content\n")
    restore_logging.append("causalts_test_log_b")
    logger = _setup(
        "causalts_test_log_b", tmp_path, console_log_output="STDERR", mode="w"
    )
    logger.error("boom")
    for h in logger.handlers:
        h.flush()
    assert "[ERROR] boom" in capsys.readouterr().err
    assert (tmp_path / "run.log").read_text() == "[ERROR] boom\n"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"console_log_output": "file"}, "invalid output: 'file'"),
        ({"console_log_level": "loud"}, "invalid level: 'loud'"),
        ({"logfile_log_level": "loud"}, "Failed to set log file log level"),
    ],
)
def test_set_up_logging_invalid_settings_return_false(
    tmp_path, capsys, restore_logging, overrides, message
):
    restore_logging.append("causalts_test_log_c")
    assert _setup("causalts_test_log_c", tmp_path, **overrides) is False
    assert message in capsys.readouterr().out


def test_set_up_logging_bad_logfile_level_closes_the_file(
    tmp_path, monkeypatch, restore_logging
):
    opened = []

    class RecordingHandler(logging.FileHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

    monkeypatch.setattr(logging, "FileHandler", RecordingHandler)
    restore_logging.append("causalts_test_log_e")
    assert _setup("causalts_test_log_e", tmp_path, logfile_log_level="loud") is False
    assert len(opened) == 1
    assert opened[0].stream is None  # closed, not left open
    assert logging.getLogger("causalts_test_log_e").handlers == []


def test_set_up_logging_unwritable_logfile_returns_false(
    tmp_path, capsys, restore_logging
):
    restore_logging.append("causalts_test_log_d")
    bad = tmp_path / "missing_dir" / "run.log"
    assert _setup("causalts_test_log_d", tmp_path, logfile_file=str(bad)) is False
    assert "Failed to set up log file" in capsys.readouterr().out
    # A failed setup leaves no console handler behind.
    assert logging.getLogger("causalts_test_log_d").handlers == []
