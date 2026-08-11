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


def read_dataframe(path: str) -> pd.DataFrame:
    """Read a tabular time series from ``path``, dispatching on file extension.

    Supported: ``.csv``, ``.parquet`` / ``.pq``, ``.feather``.  Parquet and
    feather require ``pyarrow`` (``pip install causalts[parquet]``).

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

    if ext == ".csv":
        return pd.read_csv(path)

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
        "Supported: .csv, .parquet, .pq, .feather"
    )
