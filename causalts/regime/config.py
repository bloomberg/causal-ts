# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Configuration dataclasses for regime-conditional discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegimeParams:
    """Parameters for causal discovery within a single regime."""

    max_lag: int = 5
    alpha: float = 0.01
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeConfig:
    """Full configuration for regime-conditional discovery.

    ``segment_threshold`` is the share of a regime's segments that must find
    an edge for the regime graph to keep it. ``threshold_rounding`` sets how
    that share becomes a vote count over ``n`` segments:

    - ``"floor"`` (default): ``max(1, int(segment_threshold * n))``, so at 0.5
      one segment of three suffices.
    - ``"ceil"``: the fewest votes ``k`` with ``k / n >= segment_threshold``,
      a share of at least ``segment_threshold`` (two of three at 0.5; a tie
      such as two of four still passes).
    """

    n_regimes: int = 3
    regime_params: list[RegimeParams] | None = None
    min_segment_length: int | None = None
    buffer: int = 0
    aggregation: str = "or"
    segment_threshold: float = 0.5
    regime_names: list[str] | None = None
    threshold_rounding: str = "floor"

    def __post_init__(self):
        if self.threshold_rounding not in ("floor", "ceil"):
            raise ValueError(
                "threshold_rounding must be 'floor' or 'ceil', got "
                f"{self.threshold_rounding!r}"
            )
