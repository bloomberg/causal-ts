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
    """Full configuration for regime-conditional discovery."""

    n_regimes: int = 3
    regime_params: list[RegimeParams] | None = None
    min_segment_length: int | None = None
    buffer: int = 0
    aggregation: str = "or"
    segment_threshold: float = 0.5
    regime_names: list[str] | None = None
