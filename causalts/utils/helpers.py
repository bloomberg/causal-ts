# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import sys

import numpy as np


def evaluate_graph(
    G_est: np.ndarray, G_true: np.ndarray, exclude_self_loops: bool = False
) -> dict:
    """Compare estimated graph to ground truth.

    Both inputs: [cause, effect, lag] with lag 0..max_lag.
    Self-loops at lag 0 are always excluded from evaluation.

    Parameters
    ----------
    G_est : ndarray, shape (d, d, max_lag+1)
        Estimated graph.
    G_true : ndarray, shape (d, d, max_lag+1)
        Ground truth graph.
    exclude_self_loops : bool, optional
        If *True*, exclude self-loops (autodependencies) at **all** lags from
        per-lag metrics, making them directly comparable with pair-level
        metrics. Default *False* (only lag-0 self-loops excluded).

    Returns
    -------
    dict
        Dict with TPR, FPR, Precision, F1, SHD.
    """
    G_est = np.asarray(G_est, dtype=np.int8)
    G_true = np.asarray(G_true, dtype=np.int8)
    assert (
        G_est.shape == G_true.shape
    ), f"Shape mismatch: {G_est.shape} vs {G_true.shape}"

    d = G_est.shape[0]
    # Build mask excluding self-loops
    mask = np.ones(G_est.shape, dtype=bool)
    if exclude_self_loops:
        for i in range(d):
            mask[i, i, :] = False
    else:
        for i in range(d):
            mask[i, i, 0] = False

    est_flat = G_est[mask].astype(bool)
    true_flat = G_true[mask].astype(bool)

    tp = int(np.sum(est_flat & true_flat))
    fp = int(np.sum(est_flat & ~true_flat))
    fn = int(np.sum(~est_flat & true_flat))
    tn = int(np.sum(~est_flat & ~true_flat))

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision_val = tp / max(tp + fp, 1)
    recall = tpr
    f1 = 2 * precision_val * recall / max(precision_val + recall, 1e-12)
    shd = fp + fn  # Structural Hamming Distance

    # Pair-level metrics (collapse lags): did we get the (cause→effect) link?
    d = G_est.shape[0]
    est_pair = G_est.any(axis=2).astype(bool)
    true_pair = G_true.any(axis=2).astype(bool)
    pair_mask = np.ones((d, d), dtype=bool)
    np.fill_diagonal(pair_mask, False)
    ep = est_pair[pair_mask]
    tp_p = true_pair[pair_mask]
    pair_tp = int(np.sum(ep & tp_p))
    pair_fp = int(np.sum(ep & ~tp_p))
    pair_fn = int(np.sum(~ep & tp_p))
    pair_tn = int(np.sum(~ep & ~tp_p))
    shd_pair = pair_fp + pair_fn
    tpr_pair = pair_tp / max(pair_tp + pair_fn, 1)
    fpr_pair = pair_fp / max(pair_fp + pair_tn, 1)
    prec_pair = pair_tp / max(pair_tp + pair_fp, 1)
    rec_pair = tpr_pair
    f1_pair = 2 * prec_pair * rec_pair / max(prec_pair + rec_pair, 1e-12)

    return {
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision_val,
        "F1": f1,
        "SHD": shd,
        "SHD_pair": shd_pair,
        "F1_pair": f1_pair,
        "TPR_pair": tpr_pair,
        "FPR_pair": fpr_pair,
        "Precision_pair": prec_pair,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def precision(s_true, s1):
    """Calculate the precision metric.

    The ratio of the intersection of two sets to the size of the second set.

    Precision is defined as ``len(s_true & s1) / len(s1)``.

    Parameters
    ----------
    s_true : iterable
        The ground truth set or list of elements.
    s1 : iterable
        The predicted set or list of elements.

    Returns
    -------
    float
        The precision value. Returns 0 if the second set is empty.
    """
    s = set(s_true).intersection(s1)
    if len(s1) > 0:
        return len(s) / len(s1)
    else:
        return 0


def recall(s_true, s1):
    """Calculate the recall metric.

    Measures the proportion of relevant items (from ``s_true``) that are
    present in the predicted set (``s1``).

    Parameters
    ----------
    s_true : iterable
        The ground truth set of relevant items.
    s1 : iterable
        The predicted set of items.

    Returns
    -------
    float
        The recall value, calculated as the size of the intersection of
        ``s_true`` and ``s1`` divided by the size of ``s_true``. Returns
        ``np.nan`` if ``s_true`` is empty, as recall is undefined in that
        case.
    """
    s = set(s_true).intersection(s1)
    if len(s_true) > 0:
        return len(s) / len(s_true)
    else:
        return np.nan


def precision_recall(s1, s2):
    """Calculate the precision and recall between two sets.

    Parameters
    ----------
    s1 : set
        The first set of elements.
    s2 : set
        The second set of elements.

    Returns
    -------
    tuple
        A tuple containing precision and recall values.

        - precision (float): The ratio of the intersection of ``s1``
          and ``s2`` to the size of ``s1``.
        - recall (float): The ratio of the intersection of ``s1`` and
          ``s2`` to the size of ``s2``.

    Notes
    -----
    This function assumes the existence of ``precision`` and ``recall``
    functions that compute the respective metrics based on the provided
    sets.
    """
    return precision(s1, s2), recall(s1, s2)


def shd_2d(s1, s2, double_for_anticausal=True):
    """Compute the Structural Hamming Distance (SHD) between two adjacency matrices.

    SHD is a metric used to compare graphs by their adjacency matrices. It
    calculates the number of edges that are either missing or incorrectly
    present in the target graph. For directed graphs, an edge in the wrong
    direction counts as two mistakes: one for the incorrect direction and one
    for the missing correct direction. The ``double_for_anticausal`` parameter
    determines whether to count such mistakes as double.

    Parameters
    ----------
    s1 : np.ndarray
        The first binary adjacency matrix.
    s2 : np.ndarray
        The second binary adjacency matrix to compare against.
    double_for_anticausal : bool, optional
        If True, counts mistakes for edges in the wrong direction as
        double. If False, treats them as single mistakes. Default True.

    Returns
    -------
    int
        The Structural Hamming Distance between the two adjacency matrices.
    """
    diff = np.abs(s1 - s2)
    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff) / 2


def shd_3d(s1, s2, double_for_anticausal=True):
    """Compute the Structural Hamming Distance (SHD) between two 3D adjacency matrices.

    SHD is a metric used to compare graphs by their adjacency matrices. It
    calculates the number of edges that are either missing or incorrectly
    present in the target graph. For directed graphs, SHD can count two
    mistakes for an edge: one for the edge being in the wrong direction and
    another for the edge missing in the correct direction. The
    ``double_for_anticausal`` parameter determines whether to account for
    this double counting.

    Parameters
    ----------
    s1 : numpy.ndarray
        A 3D binary adjacency matrix representing the first graph.
    s2 : numpy.ndarray
        A 3D binary adjacency matrix representing the second graph.
    double_for_anticausal : bool
        If True, counts both mistakes for edges in the wrong direction.
        If False, adjusts the calculation to avoid double counting.

    Returns
    -------
    int
        The Structural Hamming Distance between the two graphs.
    """

    diff = np.abs(s1 - s2)
    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose((1, 0, 2))
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff) / 2


def precision_recall_3d(g_true_3d, g_est_3d):
    """Calculate precision and recall for 3D graphs.

    This function computes the precision and recall metrics for comparing
    two 3D graphs, ``g_true_3d`` (ground truth) and ``g_est_3d`` (estimated
    graph). It assumes the graphs are represented as 3D numpy arrays.

    Parameters
    ----------
    g_true_3d : numpy.ndarray
        Ground truth 3D graph represented as a numpy array. The array
        should have shape (N, N, T), where N is the number of nodes and
        T is the number of time steps.
    g_est_3d : numpy.ndarray
        Estimated 3D graph represented as a numpy array. The array should
        have the same shape as ``g_true_3d``.

    Returns
    -------
    tuple
        A tuple containing:

        - precision (float): The ratio of correctly identified links to
          the total number of links in the estimated graph.
        - recall (float): The ratio of correctly identified links to the
          total number of links in the ground truth graph.

    Notes
    -----
    The function modifies the input ``g_true_3d`` by setting the diagonal
    of the second layer (index 1) to zero, effectively ignoring self-links
    for time t-1 to t. Precision is calculated as ``nTP / nDisc``, where
    ``nTP`` is the number of true positives (correct links) and ``nDisc``
    is the total number of links in the estimated graph. Recall is
    calculated as ``nTP / L``, where ``L`` is the total number of links
    in the ground truth graph. A small value (1e-12) is added to ``nDisc``
    to avoid division by zero.
    """
    # Exclude lag-1 self-loops (autoregression) from ground truth so CEDAR,
    # which never discovers Xi(t-1)→Xi(t) by design, is not penalised for them.
    g_true_3d = g_true_3d.copy()
    np.fill_diagonal(g_true_3d[:, :, 1], 0)
    nTP = np.sum((g_est_3d + g_true_3d) == 2)  # correct link
    L = g_true_3d.sum()  # number of links
    nDisc = g_est_3d.sum()
    nDisc = nDisc + 1e-12 if nDisc == 0 else nDisc
    return nTP / nDisc, nTP / L  # precission, recall


# Logging formatter supporting colorized output
class LogFormatter(logging.Formatter):
    """Logging formatter with optional ANSI color output."""

    COLOR_CODES = {
        logging.CRITICAL: "\033[1;35m",  # bright/bold magenta
        logging.ERROR: "\033[1;31m",  # bright/bold red
        logging.WARNING: "\033[1;33m",  # bright/bold yellow
        logging.INFO: "\033[0;37m",  # white / light gray
        logging.DEBUG: "\033[1;30m",  # bright/bold black / dark gray
    }

    RESET_CODE = "\033[0m"

    def __init__(self, color, *args, **kwargs):
        super(LogFormatter, self).__init__(*args, **kwargs)
        self.color = color

    def format(self, record, *args, **kwargs):
        if self.color and record.levelno in self.COLOR_CODES:
            record.color_on = self.COLOR_CODES[record.levelno]
            record.color_off = self.RESET_CODE
        else:
            record.color_on = ""
            record.color_off = ""
        return super(LogFormatter, self).format(record, *args, **kwargs)


def set_up_logging(
    logger_name,
    console_log_output,
    console_log_level,
    console_log_color,
    logfile_file,
    logfile_log_level,
    logfile_log_color,
    log_line_template,
    mode="a",
):
    # Usage example for setting up logging:
    # set_up_logging(
    #     logger_name="my_logger",
    #     console_log_output="stdout",
    #     console_log_level="warning",
    #     console_log_color=True,
    #     logfile_file="script.log",
    #     logfile_log_level="debug",
    #     logfile_log_color=False,
    #     log_line_template=(
    #         "%(color_on)s[%(created)d] [%(threadName)s] [%(levelname)-8s] "
    #         "%(message)s%(color_off)s"
    #     )
    # )
    # Create logger
    # For simplicity, we use the root logger, i.e., call 'logging.getLogger()'
    # without a name argument. This way, we can use module methods for logging
    # throughout the script. Alternatively, export the logger using:
    # 'global logger; logger = logging.getLogger("<name>")'
    logging.root.handlers = []  # KILL ALL LOGGERS
    logger = logging.getLogger(logger_name)

    # Set global log level to 'debug' (required for handler levels to work)
    logger.setLevel(logging.DEBUG)

    # Create console handler
    console_log_output = console_log_output.lower()
    if console_log_output == "stdout":
        console_log_output = sys.stdout
    elif console_log_output == "stderr":
        console_log_output = sys.stderr
    else:
        print(f"Failed to set console output: invalid output: '{console_log_output}'")
        return False
    console_handler = logging.StreamHandler(console_log_output)

    # Set console log level
    try:
        console_handler.setLevel(
            console_log_level.upper()
        )  # only accepts uppercase level names
    except ValueError:
        print(f"Failed to set console log level: invalid level: '{console_log_level}'")
        return False

    # Create and set formatter, add console handler to logger
    console_formatter = LogFormatter(fmt=log_line_template, color=console_log_color)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create log file handler
    try:
        logfile_handler = logging.FileHandler(logfile_file, mode=mode)
    except Exception as exception:
        print(f"Failed to set up log file: {exception}")
        return False

    # Set log file log level
    try:
        logfile_handler.setLevel(
            logfile_log_level.upper()
        )  # only accepts uppercase level names
    except ValueError:
        print(f"Failed to set log file log level: invalid level: '{logfile_log_level}'")
        return False

    # Create and set formatter, add log file handler to logger
    logfile_formatter = LogFormatter(fmt=log_line_template, color=logfile_log_color)
    logfile_handler.setFormatter(logfile_formatter)
    logger.addHandler(logfile_handler)

    # Success
    return logger
