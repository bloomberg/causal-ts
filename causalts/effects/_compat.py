# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

# The floor exists to produce a readable error, not to gate behaviour: the two
# APIs that moved between releases are bound by capability check in
# `graph_bridge`, not by version. See `_adjustment_api` there.
_MIN_VERSION = (0, 11)


def require_dowhy(feature: str = "this feature") -> None:
    try:
        import dowhy
    except ImportError:
        raise ImportError(
            f"{feature} requires DoWhy (>={'.'.join(map(str, _MIN_VERSION))}). "
            "Install it with: pip install dowhy"
        ) from None

    ver = tuple(int(x) for x in dowhy.__version__.split(".")[:2])
    if ver < _MIN_VERSION:
        raise ImportError(
            f"{feature} requires DoWhy >={'.'.join(map(str, _MIN_VERSION))}, "
            f"but {dowhy.__version__} is installed."
        )


def require_gcm(feature: str = "this feature") -> None:
    require_dowhy(feature)
    try:
        import dowhy.gcm  # noqa: F401
    except ImportError:
        raise ImportError(
            f"{feature} requires the DoWhy GCM module. "
            "Upgrade with: pip install 'dowhy>=0.11'"
        ) from None


def dowhy_available() -> bool:
    try:
        import dowhy  # noqa: F401

        return True
    except ImportError:
        return False


def require_identification(feature: str = "this feature") -> None:
    """Guard DoWhy's classic ``identify_effect`` path.

    Separate from :func:`require_dowhy` because the constraint is narrower than
    the package: DoWhy's ``gcm`` layer (``fit_scm``, ``counterfactual``,
    ``attribute_anomaly``, ``arrow_strength``) and this package's own
    validators work fine on older DoWhy, so they must not be gated on this.

    The incompatibility is a *pair*, not a version. DoWhy below 0.13 calls
    ``networkx.algorithms.d_separated``, which networkx removed in 3.5. Either
    side alone is fine -- DoWhy 0.12 with networkx 3.4 works -- so pinning
    DoWhy alone would penalise a working install. Detect the combination and
    name both escape routes, rather than letting an ``AttributeError`` surface
    from inside DoWhy.
    """
    require_dowhy(feature)

    import networkx as nx

    if hasattr(nx.algorithms, "d_separated"):
        return  # old networkx: every DoWhy version we support is fine

    # Diagnostic only -- never let the check itself become the failure. If the
    # version is unparsable we say nothing and let DoWhy raise on its own.
    try:
        import dowhy

        parts = []
        for chunk in dowhy.__version__.split(".")[:2]:
            digits = "".join(c for c in chunk if c.isdigit())
            if not digits:
                return
            parts.append(int(digits))
        if tuple(parts) >= (0, 13):
            return
        installed = dowhy.__version__
    except Exception:  # pragma: no cover - defensive
        return

    raise ImportError(
        f"{feature} needs DoWhy's identification path, which is broken by the "
        f"combination of DoWhy {installed} and networkx {nx.__version__}: DoWhy "
        "<0.13 calls networkx.algorithms.d_separated, removed in networkx 3.5. "
        "Fix either side -- 'pip install dowhy>=0.13' (recommended), or pin "
        "'networkx<3.5'. DoWhy's gcm features (fit_scm, counterfactual, "
        "attribute_anomaly) are unaffected and need no change."
    )
