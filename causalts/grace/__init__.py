# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

from ..algorithms import register_algorithm
from .gated_discovery import (
    GatedCausalDiscovery,
    StabilitySelector,
    background_xcorr,
    compute_lambda,
    compute_lambda_cv,
    evaluate_graph,
    gates_bimodal,
    li_threshold,
    prepare_data,
    run_cdnots_gated,
    run_stability_selection,
)
from .result import GraceResult  # noqa: F401
from .runners import run_grace, run_reap  # noqa: F401
from .shrink import shrink_parents  # noqa: F401

register_algorithm("grace")(run_cdnots_gated)
register_algorithm("grace-ss")(run_stability_selection)
register_algorithm("reap")(run_reap)
