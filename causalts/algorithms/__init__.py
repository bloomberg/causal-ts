# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Algorithm plugin registry for causal-ts.

Third-party algorithms register a discovery function under a short name and then
work anywhere a built-in does -- :func:`run_algorithm`, ``list_algorithms()``, and
``causal-ts discover --algorithm <name>``.

The function must accept the calling convention :func:`run_algorithm` uses --
``fn(df=..., ci_test=..., max_lag=..., **kwargs)`` -- and return a
:class:`~causalts.result.CausalResult` subclass::

    from causalts import register_algorithm, CausalResult

    @register_algorithm("my_algo")
    def run_my_algo(df, ci_test, max_lag, **kwargs):
        ...
        return MyResult(graph, df, list(df.columns))

``@register_algorithm`` only registers within the running process, so it covers
the Python API and any script that imports your module. To reach the installed
``causal-ts`` command -- a separate process that never imports your code -- also
advertise it in your own distribution's ``pyproject.toml``::

    [project.entry-points."causalts.algorithms"]
    my_algo = "my_package.plugin:run_my_algo"

Entry points are discovered on first use (see
:func:`_load_entry_point_algorithms`), so an installed plugin needs no import on
the caller's side.

**CLI limitations.** ``causal-ts discover --algorithm <plugin>`` forwards only
``df``/``ci_test``/``max_lag`` -- CDNOTS/CEDAR/GRACE-specific flags like
``--alpha``, ``--include-c``, or ``--max-degree`` are parsed but not passed
through, since they have no defined meaning for an arbitrary plugin. A plugin
that needs more configuration should read it from its own environment
variable, config file, or a separate CLI, and document that. ``--validate``
is also not supported for plugins -- the stability bootstrap only knows how
to re-run the built-in algorithms, so it is rejected outright rather than
silently reporting another algorithm's persistence values.
"""

from __future__ import annotations

import warnings

_ALGO_REGISTRY: dict[str, callable] = {}

#: Entry-point group third-party distributions advertise algorithms under.
ENTRY_POINT_GROUP = "causalts.algorithms"

_entry_points_loaded = False

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


def _load_entry_point_algorithms() -> None:
    """Register algorithms advertised under the :data:`ENTRY_POINT_GROUP`.

    Lets an installed third-party distribution reach the ``causal-ts`` command,
    which runs in its own process and never imports the caller's modules.  Runs
    once, on first use, so plain ``import causalts`` pays nothing until an
    algorithm is actually looked up.

    A plugin that fails to import is warned about and skipped -- one broken
    third-party package must not make the CLI unusable.  Plugins never shadow a
    built-in name.
    """
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True  # set first: a raising plugin must not retry forever

    from importlib.metadata import entry_points

    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"could not scan {ENTRY_POINT_GROUP!r} entry points: {exc}")
        return

    for ep in eps:
        if ep.name in _ALGO_REGISTRY or ep.name in _LAZY_ALGO_REGISTRY:
            warnings.warn(
                f"algorithm plugin {ep.name!r} clashes with a built-in name and "
                f"was ignored"
            )
            continue
        try:
            _ALGO_REGISTRY[ep.name] = ep.load()
        except Exception as exc:
            warnings.warn(f"could not load algorithm plugin {ep.name!r}: {exc}")


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
    """Return sorted list of all registered algorithm names.

    Includes installed entry-point plugins.  ``causalts.cli`` calls this at import
    time to build ``--algorithm``'s choices, so plugins appear there too.
    """
    _load_entry_point_algorithms()
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
            _load_entry_point_algorithms()
    if name not in _ALGO_REGISTRY:
        raise ValueError(f"Unknown algorithm {name!r}. Available: {list_algorithms()}")
    return _ALGO_REGISTRY[name](df=df, ci_test=ci_test, max_lag=max_lag, **kwargs)
