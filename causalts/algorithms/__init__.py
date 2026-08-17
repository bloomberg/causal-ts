# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Algorithm plugin registry for causal-ts.

Third-party algorithms can register themselves with :func:`register_algorithm`
so they are automatically available in the CLI without editing the package::

    from causalts import register_algorithm, CausalResult

    @register_algorithm("my_algo")
    def run_my_algo(df, ci_test, max_lag, **kwargs):
        ...
        return MyResult(cg_tig=..., var_names=..., _df=df)

After the import, ``causal-ts discover --algorithm my_algo data.csv`` works.
"""

from __future__ import annotations

_ALGO_REGISTRY: dict[str, callable] = {}

# Algorithms whose module is expensive to import (GRACE pulls in
# pytorch-lightning) are registered lazily: name -> (module path, function).
# ``causalts.grace`` used to self-register at import time; registering here
# keeps the names listed without paying for the import.  Mirrors the
# _LAZY_REGISTRY pattern in causalts.ci_tests.
_LAZY_ALGO_REGISTRY: dict[str, tuple[str, str]] = {
    "grace": ("causalts.grace.gated_discovery", "run_cdnots_gated"),
    "grace-ss": ("causalts.grace.gated_discovery", "run_stability_selection"),
}


def _resolve_lazy(name: str):
    """Import a lazily-registered algorithm and cache it in the registry."""
    import importlib

    module_path, fn_name = _LAZY_ALGO_REGISTRY[name]
    fn = getattr(importlib.import_module(module_path), fn_name)
    _ALGO_REGISTRY[name] = fn
    return fn


def register_algorithm(name: str):
    """Decorator to register a discovery function under *name*.

    Parameters
    ----------
    name : str
        Short identifier used in ``--algorithm`` CLI option and
        :func:`run_algorithm`.

    Returns
    -------
    decorator
        Passes the function through unchanged; side-effect only.
    """

    def decorator(fn):
        _ALGO_REGISTRY[name] = fn
        return fn

    return decorator


def list_algorithms() -> list[str]:
    """Return sorted list of all registered algorithm names."""
    return sorted(set(_ALGO_REGISTRY) | set(_LAZY_ALGO_REGISTRY))


def run_algorithm(name: str, df, ci_test, max_lag, **kwargs):
    """Invoke a registered algorithm by name.

    Parameters
    ----------
    name : str
        Algorithm name as registered with :func:`register_algorithm`.
    df : pd.DataFrame
        Time series data.
    ci_test : CIT_Base
        Conditional independence test instance.
    max_lag : int
        Maximum lag.
    **kwargs
        Forwarded to the algorithm function.

    Returns
    -------
    CausalResult
    """
    if name not in _ALGO_REGISTRY:
        if name in _LAZY_ALGO_REGISTRY:
            _resolve_lazy(name)
        else:
            raise ValueError(
                f"Unknown algorithm {name!r}. Available: {list_algorithms()}"
            )
    return _ALGO_REGISTRY[name](df=df, ci_test=ci_test, max_lag=max_lag, **kwargs)
