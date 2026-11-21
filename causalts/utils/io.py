# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Multi-format table reader for the CLI.

A single dispatch point so the file-based CLI (``inspect``, ``discover``,
``deconfound``) accepts the same formats regardless of command, and errors
actionably when an optional engine is missing.
"""

from __future__ import annotations

import os

import pandas as pd

_PARQUET_EXTS = (".parquet", ".pq")


class IndexColumnError(ValueError):
    """:func:`read_dataframe` was given an ``index_col`` the file does not have."""


def read_dataframe(path: str, index_col: str | int | None = None) -> pd.DataFrame:
    """Read a tabular time series from ``path``, dispatching on file extension.

    Supported: ``.csv``, ``.parquet`` / ``.pq``, ``.feather``.  Parquet and
    feather require ``pyarrow`` (``pip install causalts[parquet]``).

    Parameters
    ----------
    path : str
        Path to the data file.
    index_col : str or int, optional
        Column to read as the row index (e.g. a timestamp column), so that it is
        not one of the variables. A column name, or a 0-based position; a name
        is matched first, so a column called ``"0"`` wins over position 0.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If the extension is not recognised.
    IndexColumnError
        If ``index_col`` is neither a column name nor a valid position. Subclass
        of ``ValueError``.
    ImportError
        If a parquet/feather engine (pyarrow) is required but not installed.
    """
    df = _read_table(path)
    if index_col is not None:
        df = df.set_index(_index_label(df, index_col))
    return df


def _read_table(path: str) -> pd.DataFrame:
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


def _index_label(df: pd.DataFrame, index_col: str | int):
    """The column label ``index_col`` refers to: a name first, then a position."""
    names = [str(c) for c in df.columns]
    key = str(index_col)
    if key in names:
        return df.columns[names.index(key)]
    if key.isdecimal() and int(key) < len(names):
        return df.columns[int(key)]
    raise IndexColumnError(
        f"index column {index_col!r} is neither a column name nor a 0-based "
        f"position; the columns are {names}."
    )
