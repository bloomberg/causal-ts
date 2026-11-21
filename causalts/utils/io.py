# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Multi-format table reader for the CLI.

A single dispatch point so the file-based CLI (``inspect``, ``discover``) accepts
the same formats regardless of command, and errors actionably when an optional
engine is missing.
"""

from __future__ import annotations

import os

import pandas as pd

_PARQUET_EXTS = (".parquet", ".pq")

# Compression suffixes stripped only to choose the CSV/TSV reader. The
# original path is always passed to pandas, which infers the compression.
_COMPRESSION_EXTS = (".gz", ".bz2", ".zip", ".xz", ".zst")


def _base_ext(path: str) -> str:
    """Extension of ``path`` ignoring a trailing compression suffix."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _COMPRESSION_EXTS:
        ext = os.path.splitext(os.path.splitext(path)[0])[1].lower()
    return ext


def read_dataframe(path: str) -> pd.DataFrame:
    """Read a tabular time series from ``path``, dispatching on file extension.

    Supported: ``.csv`` (including compressed ``.csv.gz``, ``.csv.bz2``,
    ``.csv.zip``, ``.csv.xz`` and ``.csv.zst``), ``.tsv`` (tab-separated),
    ``.parquet`` / ``.pq``, ``.feather``.  Parquet and feather require
    ``pyarrow`` (``pip install causalts[parquet]``); ``.zst`` requires the
    optional ``zstandard`` package.  A ``.zip`` must contain a single table.
    A compression suffix on a non-CSV/TSV file (e.g. ``.parquet.gz``) is not
    supported.

    Parameters
    ----------
    path : str
        Path to the data file.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If the extension is not recognised.
    ImportError
        If a parquet/feather engine (pyarrow) is required but not installed.
    """
    ext = os.path.splitext(path)[1].lower()
    base_ext = _base_ext(path)

    if base_ext == ".csv":
        return pd.read_csv(path)

    if base_ext == ".tsv":
        return pd.read_csv(path, sep="\t")

    if ext in _PARQUET_EXTS:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "Reading parquet requires a parquet engine. "
                "Install it with: pip install causalts[parquet]"
            ) from exc

    if ext == ".feather":
        try:
            return pd.read_feather(path)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "Reading feather requires pyarrow. "
                "Install it with: pip install causalts[parquet]"
            ) from exc

    raise ValueError(
        f"Unsupported file format '{ext}' for '{path}'. "
        "Supported: .csv (.csv.gz, .csv.bz2, .csv.zip, .csv.xz, .csv.zst), "
        ".tsv, .parquet, .pq, .feather"
    )
