# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

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
            f"{feature} requires the DoWhy GCM module (available in DoWhy >=0.11). "
            "Upgrade with: pip install 'dowhy>=0.11'"
        ) from None


def dowhy_available() -> bool:
    try:
        import dowhy  # noqa: F401

        return True
    except ImportError:
        return False
