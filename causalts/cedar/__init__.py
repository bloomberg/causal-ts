# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import TYPE_CHECKING

from .discovery import Cedar, run_cedar  # noqa: F401
from .result import CedarResult  # noqa: F401

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from .legacy import SYPI  # noqa: F401


def __getattr__(name):
    """Serve the deprecated SYPI shim, and its module, on demand.

    ``legacy`` imports dcor + statsmodels, which is a large share of the cost of
    ``import causalts``; nothing on the Cedar path needs it.
    """
    import importlib

    if name == "legacy":
        value = importlib.import_module(f"{__name__}.legacy")
    elif name == "SYPI":  # deprecated — use run_cedar / Cedar
        value = importlib.import_module(f"{__name__}.legacy").SYPI
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)


__all__ = sorted(
    ({name for name in globals() if not name.startswith("_")} - {"TYPE_CHECKING"})
    | {"SYPI", "legacy"}
)
