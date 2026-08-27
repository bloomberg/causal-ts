# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""LUCID: regime-adaptive deconfounding for time-series causal discovery.

Latent confounders leave different statistical fingerprints depending on their
structure, and no single correction handles all of them. LUCID infers the confounding
regime from the data with a Marchenko-Pastur spectral router, then applies the strategy
matched to that regime.

The main entry point is :func:`run_lucid`, which returns a
:class:`~causalts.confounders.result.LucidResult` carrying the graph alongside the
router's diagnostics. :func:`routed_deconfound` is the lower-level array-returning form.
Every result object also exposes ``.deconfound()``, which applies LUCID to an
already-discovered graph without re-running the skeleton search.

See :mod:`causalts.confounders.routed_deconf` for details.
"""

from .result import LucidResult
from .routed_deconf import (
    deconfound,
    mp_factor_count,
    pds_filter,
    routed_deconfound,
    run_lucid,
    spectral_gap,
    tetrad_filter,
)

__all__ = [
    "run_lucid",
    "LucidResult",
    "routed_deconfound",
    "deconfound",
    "spectral_gap",
    "mp_factor_count",
    "pds_filter",
    "tetrad_filter",
]
