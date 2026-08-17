# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("causalts")
except PackageNotFoundError:
    __version__ = "0.14.0-dev"

from .algorithms import list_algorithms, register_algorithm  # noqa: F401
from .bootstrap import temporal_bootstrap  # noqa: F401
from .cdnots.ancestral import AncestralKnowledge  # noqa: F401
from .cdnots.phase3_utils import make_c_array  # noqa: F401
from .cdnots.phase3_utils import run_cdnots  # noqa: F401
from .cdnots.phase3_utils import run_cdnots_plus  # noqa: F401
from .cdnots.phase3_utils import (  # deprecated — use run_cdnots  # noqa: F401; noqa: F401
    cdnots_discovery,
)
from .cdnots.result import CdnotsResult  # noqa: F401
from .cedar.discovery import Cedar, run_cedar  # noqa: F401
from .cedar.result import CedarResult  # noqa: F401
from .feature_selection import (  # noqa: F401
    Feature,
    FeatureSelectionResult,
    select_features,
)
from .inspection import discover_df, inspect_df, recommend_config  # noqa: F401
from .result import CausalResult  # noqa: F401
from .tigramite_discovery import (  # noqa: F401
    run_lpcmci,
    run_pcmci,
    run_pcmciplus,
    run_tigramite,
)

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from .cedar.legacy import SYPI  # noqa: F401
    from .grace.gated_discovery import (  # noqa: F401
        run_cdnots_gated,
        run_stability_selection,
    )
    from .grace.result import GraceResult  # noqa: F401

# Served on demand so that ``import causalts`` stays cheap: cedar.legacy pulls
# in dcor + statsmodels, grace pulls in pytorch-lightning, and neither is on the
# common discovery path.  Maps attribute name -> module that defines it.
_LAZY_ATTRS = {
    "SYPI": "causalts.cedar.legacy",
    "run_cdnots_gated": "causalts.grace.gated_discovery",
    "run_stability_selection": "causalts.grace.gated_discovery",
    "GraceResult": "causalts.grace.result",
}

# Subpackages that used to be bound as attributes of ``causalts`` as a side
# effect of the eager imports above.  Keeps ``causalts.grace`` and
# ``causalts.plotting`` resolving without importing them up front.
_LAZY_SUBMODULES = ("grace", "plotting")


def __getattr__(name):
    """Import a lazily-exposed attribute or subpackage on first access (PEP 562)."""
    import importlib

    if name in _LAZY_SUBMODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    elif name in _LAZY_ATTRS:
        value = getattr(importlib.import_module(_LAZY_ATTRS[name]), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # cache so __getattr__ only runs once per name
    return value


def __dir__():
    return sorted(__all__)


# Explicit __all__ so that ``from causalts import *`` still exports the lazy
# names; star-import consults __all__ and never triggers __getattr__ without it.
__all__ = sorted(
    ({name for name in globals() if not name.startswith("_")} - {"TYPE_CHECKING"})
    | set(_LAZY_ATTRS)
    | set(_LAZY_SUBMODULES)
)
