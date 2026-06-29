# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

from importlib.metadata import PackageNotFoundError, version

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
from .cedar.legacy import SYPI  # noqa: F401
from .cedar.result import CedarResult  # noqa: F401
from .grace.gated_discovery import (  # noqa: F401
    run_cdnots_gated,
    run_stability_selection,
)
from .grace.result import GraceResult  # noqa: F401
from .result import CausalResult  # noqa: F401
from .tigramite_discovery import (  # noqa: F401
    run_lpcmci,
    run_pcmci,
    run_pcmciplus,
    run_tigramite,
)
